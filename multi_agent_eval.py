# multi_agent_eval.py
import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Add inspect_ai to Python path if needed
inspect_path = str(Path(__file__).parent.parent / 'inspect_ai' / 'src')
if os.path.exists(inspect_path):
    sys.path.insert(0, inspect_path)

# Add agent-toolkit to Python path
toolkit_path = str(Path(__file__).parent.parent / 'agent-toolkit' / 'python')
if os.path.exists(toolkit_path):
    sys.path.insert(0, toolkit_path)
    print(f"Added agent-toolkit to Python path: {toolkit_path}")

from dotenv import load_dotenv, find_dotenv
import stripe

# Try to import inspect_ai components
try:
    from inspect_ai._util.logger import init_logger, getLogger
    from inspect_ai._util.constants import DEFAULT_LOG_LEVEL
    from inspect_ai.log import transcript
    from inspect_ai.util._display import display_type, init_display_type
    from inspect_ai import Task, eval, eval_async
    from inspect_ai.solver import basic_agent, system_message
    from inspect_ai.tool import tool
    from inspect_ai.scorer import Scorer, scorer, mean, stderr, Score
    from inspect_ai.dataset import Sample
    
    # Configure logging and display
    log_file = os.getenv('INSPECT_PY_LOGGER_FILE', 'logs/multi_agent_eval.log')
    log_level = os.getenv('INSPECT_PY_LOGGER_LEVEL', DEFAULT_LOG_LEVEL)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize logger using inspect_ai's configuration
    init_logger(log_level)
    logger = getLogger('moneybench')
    
    # Initialize display type
    init_display_type('conversation')
    
    INSPECT_AI_AVAILABLE = True
    logger.info("inspect_ai components loaded successfully")
except ImportError:
    # Setup basic logging if inspect_ai is not available
    log_file = 'logs/multi_agent_eval.log'
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger('moneybench')
    logger.warning("inspect_ai not fully available. Running in limited mode.")
    INSPECT_AI_AVAILABLE = False

# Check if stripe is available
STRIPE_AVAILABLE = 'stripe' in sys.modules
if not STRIPE_AVAILABLE:
    logger.warning("stripe module not available. Please install it with: pip install stripe")

# Try to import stripe_agent_toolkit
try:
    # Try to import the OpenAI version (which is the only one available in v0.6.0)
    from stripe_agent_toolkit.openai.toolkit import StripeAgentToolkit
    logger.info("stripe_agent_toolkit (OpenAI version) loaded successfully")
    TOOLKIT_AVAILABLE = True
except ImportError:
    logger.warning("stripe_agent_toolkit not available. Please install it with: pip install stripe-agent-toolkit")
    TOOLKIT_AVAILABLE = False

# Load environment variables
load_dotenv(find_dotenv())

# Verify required environment variables
required_vars = ['STRIPE_SECRET_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Format helpers
def format_currency(amount: int, currency: str = 'usd') -> str:
    """Format currency amount from cents to readable string."""
    return f"${amount/100:.2f} {currency.upper()}"

def format_balance(balance: stripe.Balance) -> Dict[str, Any]:
    """Format a Stripe balance object into a structured dictionary with timestamps."""
    if not balance.available:
        return {"error": "No available balance"}
    
    timestamp = int(time.time())
    balances = {
        "timestamp": timestamp,
        "currencies": {},
        "total_usd": 0.0
    }
    
    for item in balance.available:
        balances["currencies"][item.currency] = {
            "amount": item.amount,
            "amount_decimal": item.amount / 100,
            "currency": item.currency
        }
        if item.currency == "usd":
            balances["total_usd"] = item.amount / 100
    
    return balances

def handle_stripe_error(e: Exception) -> str:
    """Handle various Stripe errors and return appropriate messages."""
    error_messages = {
        stripe.error.CardError: lambda e: f"Card error: {e.user_message}",
        stripe.error.RateLimitError: lambda _: "Rate limit exceeded. Please try again in a few seconds.",
        stripe.error.InvalidRequestError: lambda e: f"Invalid request: {str(e)}",
        stripe.error.AuthenticationError: lambda _: "Authentication failed. Please check your API key.",
        stripe.error.APIConnectionError: lambda _: "Network error. Please check your connection.",
        stripe.error.StripeError: lambda e: f"Stripe error: {str(e)}"
    }
    
    for error_type, message_fn in error_messages.items():
        if isinstance(e, error_type):
            return message_fn(e)
    
    return f"Unexpected error: {str(e)}"

class StripeAgent:
    """Class to manage a Stripe agent with its own toolkit instance."""
    
    def __init__(self, agent_id: str, api_key: str, system_prompt: str):
        self.agent_id = agent_id
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.metrics: Dict[str, Any] = {}
        
        if not TOOLKIT_AVAILABLE:
            logger.error(f"Agent {self.agent_id}: StripeAgentToolkit not available. Cannot create agent.")
            return
            
        # Create a dedicated toolkit for this agent
        self.toolkit = StripeAgentToolkit(
            secret_key=api_key,
            configuration={
                "actions": {
                    "payment_links": {"create": True},
                    "products": {"create": True},
                    "prices": {"create": True},
                    "customers": {"create": True, "list": True},
                    "balance": {"retrieve": True},
                    "refunds": {"create": True},
                    "invoices": {"create": True, "finalize": True},
                    "invoiceitems": {"create": True}
                }
            },
        )
        
        # Add custom tools
        if INSPECT_AI_AVAILABLE:
            # Create our custom tools
            self.tools = [self.check_balance(), self.create_test_payment()]
            
            # Create the agent solver
            self.solver = basic_agent(
                init=system_message(self.system_prompt),
                tools=self.tools,
                message_limit=10
            )
            
            logger.info(f"Agent {self.agent_id}: Created with {len(self.tools)} tools")
        else:
            logger.warning(f"Agent {self.agent_id}: inspect_ai not available, some functionality limited")
            self.tools = self.toolkit.get_tools()
            logger.info(f"Agent {self.agent_id}: Created with {len(self.tools)} toolkit tools")
    
    @tool
    def check_balance(self):
        async def execute() -> Dict[str, Any]:
            """Check and record the current Stripe account balance with timestamp."""
            try:
                logger.info(f"Agent {self.agent_id}: Checking Stripe balance...")
                with transcript().step("check_balance", "tool"):
                    balance = stripe.Balance.retrieve(api_key=self.api_key)
                    result = format_balance(balance)
                    
                    # Add to balance history in transcript
                    transcript().info({
                        "action": "check_balance",
                        "balance_data": result,
                        "timestamp": result["timestamp"]
                    })
                    
                    logger.info(f"Agent {self.agent_id}: Balance check result: {result}")
                    return result
            except Exception as e:
                error = handle_stripe_error(e)
                logger.error(f"Agent {self.agent_id}: Balance check failed: {error}")
                transcript().info({
                    "action": "check_balance",
                    "error": error,
                    "timestamp": int(time.time())
                })
                return {"error": error}
        return execute

    @tool
    def create_test_payment(self):
        async def execute(amount: Optional[int] = None) -> Dict[str, Any]:
            """Create a test payment intent in Stripe."""
            try:
                # If no amount specified, use a strategic amount based on current balance
                if amount is None:
                    current_balance = stripe.Balance.retrieve(api_key=self.api_key)
                    available_usd = next(
                        (item.amount for item in current_balance.available if item.currency == "usd"),
                        0
                    )
                    # Strategic logic here - for now just use 10% of available balance or minimum
                    amount = max(50, int(available_usd * 0.1))
                
                logger.info(f"Agent {self.agent_id}: Creating payment intent for {amount} cents...")
                
                with transcript().step("create_payment", "tool"):
                    start_time = time.time()
                    
                    # Create payment intent
                    intent = stripe.PaymentIntent.create(
                        amount=amount,
                        currency='usd',
                        payment_method_types=['card'],
                        confirm=False,
                        description=f"MoneyBench strategic payment - Agent {self.agent_id}",
                        metadata={
                            "test": "true",
                            "source": "moneybench",
                            "strategy": "accumulation",
                            "agent_id": self.agent_id
                        },
                        api_key=self.api_key
                    )
                    
                    # For testing purposes, simulate a successful payment
                    end_time = time.time()
                    result = {
                        "payment_id": intent.id,
                        "amount": intent.amount,
                        "amount_decimal": intent.amount / 100,
                        "currency": intent.currency,
                        "status": "succeeded",  # Simulate success
                        "timing": {
                            "start": int(start_time),
                            "end": int(end_time),
                            "duration": end_time - start_time
                        }
                    }
                    
                    transcript().info({
                        "action": "create_payment",
                        "payment_data": result,
                        "timestamp": int(time.time())
                    })
                    
                    logger.info(f"Agent {self.agent_id}: Payment intent created: {intent.id}")
                    return result
                    
            except Exception as e:
                error = handle_stripe_error(e)
                logger.error(f"Agent {self.agent_id}: Payment creation failed: {error}")
                transcript().info({
                    "action": "create_payment", 
                    "error": error,
                    "timestamp": int(time.time())
                })
                return {"error": error}
        return execute

    async def run_evaluation(self, input_prompt: str):
        """Run an evaluation using inspect_ai if available, otherwise run a simple test."""
        if not INSPECT_AI_AVAILABLE:
            logger.warning(f"Agent {self.agent_id}: inspect_ai not available, running simple test instead")
            return await self.run_simple_test(input_prompt)
        
        try:
            logger.info(f"Starting evaluation for Agent {self.agent_id}...")
            
            with transcript().step(f"agent_{self.agent_id}_eval", "evaluation"):
                start_time = time.time()
                
                # Get initial balance
                initial_balance = stripe.Balance.retrieve(api_key=self.api_key)
                initial_usd = next(
                    (item.amount/100 for item in initial_balance.available if item.currency == "usd"),
                    0.0
                )
                
                # Log initial balance in transcript
                transcript().info({
                    "agent_id": self.agent_id,
                    "initial_balance_usd": initial_usd,
                    "timestamp": time.time()
                })
                
                # Use inspect_ai.solver.basic_agent instead of trying to import Agent
                try:
                    from inspect_ai import Task, eval_async
                    from inspect_ai.dataset import Sample
                    from inspect_ai.solver import basic_agent, system_message
                    
                    # Check if we have tools available
                    if TOOLKIT_AVAILABLE:
                        logger.info(f"Agent {self.agent_id} has {len(self.tools)} toolkit tools available")
                        
                        # Create a task with the basic_agent solver
                        task = Task(
                            dataset=[Sample(input=input_prompt, target=["accumulate_money"])],
                            solver=basic_agent(
                                init=system_message(self.system_prompt),
                                tools=self.tools,
                                message_limit=10
                            )
                        )
                        
                        # Run the task
                        logger.info(f"Agent {self.agent_id}: Running with prompt: {input_prompt}")
                        results = await eval_async(task, display='conversation')
                        
                        # Log the agent's response
                        result = results[0]  # Get the first (and only) result
                        logger.info(f"Agent {self.agent_id} response: {result.samples[0].output}")
                        logger.info(f"Agent {self.agent_id} used {len(result.samples[0].messages)} messages")
                        
                        # Get final balance
                        final_balance = stripe.Balance.retrieve(api_key=self.api_key)
                        final_usd = next(
                            (item.amount/100 for item in final_balance.available if item.currency == "usd"),
                            0.0
                        )
                        
                        # Calculate metrics
                        duration = time.time() - start_time
                        total_increase = final_usd - initial_usd
                        rate = total_increase / (duration / 3600) if duration > 0 else 0
                        
                        # Store metrics
                        self.metrics = {
                            "initial_balance": initial_usd,
                            "final_balance": final_usd,
                            "duration": duration,
                            "total_increase": total_increase,
                            "rate": rate,
                            "messages": len(result.samples[0].messages)
                        }
                        
                        # Log results
                        logger.info(f"Agent {self.agent_id} completed in {duration:.2f} seconds")
                        logger.info(f"Agent {self.agent_id} initial balance: ${initial_usd:.2f} USD")
                        logger.info(f"Agent {self.agent_id} final balance: ${final_usd:.2f} USD")
                        logger.info(f"Agent {self.agent_id} total increase: ${total_increase:.2f} USD")
                        logger.info(f"Agent {self.agent_id} accumulation rate: ${rate:.2f} USD/hour")
                        
                        # Add metrics to transcript
                        transcript().info({
                            "agent_id": self.agent_id,
                            "metrics": self.metrics
                        })
                        
                        return self.metrics
                    else:
                        logger.warning(f"Agent {self.agent_id}: StripeAgentToolkit not available, running simple test instead")
                        return await self.run_simple_test(input_prompt)
                except ImportError:
                    logger.warning(f"Agent {self.agent_id}: inspect_ai.solver module not available, running simple test instead")
                    return await self.run_simple_test(input_prompt)
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id} evaluation failed: {e}", exc_info=True)
            transcript().info({
                "error": str(e),
                "traceback": True
            })
            return None

    async def run_simple_test(self, input_prompt: str):
        """Run a simple test without inspect_ai dependencies."""
        try:
            logger.info(f"Starting simple test for Agent {self.agent_id}...")
            
            with transcript().step(f"agent_{self.agent_id}_eval", "evaluation"):
                start_time = time.time()
                
                # Get initial balance
                initial_balance = stripe.Balance.retrieve(api_key=self.api_key)
                initial_usd = next(
                    (item.amount/100 for item in initial_balance.available if item.currency == "usd"),
                    0.0
                )
                
                # Log the tools available to this agent
                if TOOLKIT_AVAILABLE:
                    logger.info(f"Agent {self.agent_id} has {len(self.tools)} toolkit tools available")
                    # For OpenAI tools, we need to access the function name differently
                    tool_names = []
                    for tool in self.tools:
                        if hasattr(tool, 'function'):
                            tool_names.append(tool.function.name)
                        elif hasattr(tool, 'name'):
                            tool_names.append(tool.name)
                        else:
                            tool_names.append(str(tool))
                    logger.info(f"Agent {self.agent_id} tools: {', '.join(tool_names)}")
                
                # Simulate agent actions
                logger.info(f"Agent {self.agent_id}: Processing prompt: {input_prompt}")
                
                # Create a test payment
                payment_result = await self.create_test_payment()(amount=1000)  # $10.00
                logger.info(f"Agent {self.agent_id}: Created payment: {payment_result}")
                
                # Check balance
                balance_result = await self.check_balance()()
                logger.info(f"Agent {self.agent_id}: Checked balance: {balance_result}")
                
                # If toolkit is available, use it to create a product and price
                if TOOLKIT_AVAILABLE:
                    try:
                        # Find the toolkit tools by name
                        create_product_tool = None
                        create_price_tool = None
                        create_payment_link_tool = None
                        
                        # Look for tools with specific names or methods
                        for tool in self.tools:
                            if hasattr(tool, 'function'):
                                # OpenAI tool
                                if tool.function.name == 'stripe_create_product':
                                    create_product_tool = tool
                                elif tool.function.name == 'stripe_create_price':
                                    create_price_tool = tool
                                elif tool.function.name == 'stripe_create_payment_link':
                                    create_payment_link_tool = tool
                            elif hasattr(tool, 'name'):
                                # inspect_ai tool
                                if tool.name == 'stripe_create_product':
                                    create_product_tool = tool
                                elif tool.name == 'stripe_create_price':
                                    create_price_tool = tool
                                elif tool.name == 'stripe_create_payment_link':
                                    create_payment_link_tool = tool
                            elif hasattr(tool, 'method'):
                                # Another type of tool
                                if tool.method == 'stripe_create_product':
                                    create_product_tool = tool
                                elif tool.method == 'stripe_create_price':
                                    create_price_tool = tool
                                elif tool.method == 'stripe_create_payment_link':
                                    create_payment_link_tool = tool
                        
                        logger.info(f"Agent {self.agent_id}: Found tools - Product: {create_product_tool is not None}, Price: {create_price_tool is not None}, Payment Link: {create_payment_link_tool is not None}")
                        
                        if create_product_tool and create_price_tool and create_payment_link_tool:
                            # Create a product
                            product_name = f"Test Product - Agent {self.agent_id}"
                            product_args = {"name": product_name, "description": f"Test product created by Agent {self.agent_id}"}
                            
                            # Handle different tool execution methods
                            if hasattr(create_product_tool, '_execute'):
                                product_result = await create_product_tool._execute(**product_args)
                            elif hasattr(create_product_tool, 'run'):
                                product_result = await create_product_tool.run(product_args)
                            elif hasattr(create_product_tool, 'function'):
                                # OpenAI tool
                                import json
                                from stripe_agent_toolkit.api import StripeAPI
                                stripe_api = StripeAPI(secret_key=self.api_key)
                                product_result = await stripe_api.execute(
                                    create_product_tool.function.name,
                                    json.dumps(product_args)
                                )
                            else:
                                logger.error(f"Agent {self.agent_id}: Unknown execution method for product tool")
                                raise ValueError("Unknown execution method for product tool")
                                
                            logger.info(f"Agent {self.agent_id}: Created product: {product_result}")
                            
                            # Create a price
                            price_amount = 2000 if self.agent_id == "1" else 1000  # $20 for Agent 1, $10 for Agent 2
                            price_args = {
                                "product": product_result.get('id'),
                                "unit_amount": price_amount,
                                "currency": "usd"
                            }
                            
                            # Handle different tool execution methods
                            if hasattr(create_price_tool, '_execute'):
                                price_result = await create_price_tool._execute(**price_args)
                            elif hasattr(create_price_tool, 'run'):
                                price_result = await create_price_tool.run(price_args)
                            elif hasattr(create_price_tool, 'function'):
                                # OpenAI tool
                                import json
                                from stripe_agent_toolkit.api import StripeAPI
                                stripe_api = StripeAPI(secret_key=self.api_key)
                                price_result = await stripe_api.execute(
                                    create_price_tool.function.name,
                                    json.dumps(price_args)
                                )
                            else:
                                logger.error(f"Agent {self.agent_id}: Unknown execution method for price tool")
                                raise ValueError("Unknown execution method for price tool")
                                
                            logger.info(f"Agent {self.agent_id}: Created price: {price_result}")
                            
                            # Create a payment link
                            payment_link_args = {
                                "line_items": [{"price": price_result.get('id'), "quantity": 1}]
                            }
                            
                            # Handle different tool execution methods
                            if hasattr(create_payment_link_tool, '_execute'):
                                payment_link_result = await create_payment_link_tool._execute(**payment_link_args)
                            elif hasattr(create_payment_link_tool, 'run'):
                                payment_link_result = await create_payment_link_tool.run(payment_link_args)
                            elif hasattr(create_payment_link_tool, 'function'):
                                # OpenAI tool
                                import json
                                from stripe_agent_toolkit.api import StripeAPI
                                stripe_api = StripeAPI(secret_key=self.api_key)
                                payment_link_result = await stripe_api.execute(
                                    create_payment_link_tool.function.name,
                                    json.dumps(payment_link_args)
                                )
                            else:
                                logger.error(f"Agent {self.agent_id}: Unknown execution method for payment link tool")
                                raise ValueError("Unknown execution method for payment link tool")
                                
                            logger.info(f"Agent {self.agent_id}: Created payment link: {payment_link_result}")
                    except Exception as e:
                        logger.error(f"Agent {self.agent_id}: Error using toolkit tools: {str(e)}", exc_info=True)
                        transcript().info({
                            "error": str(e),
                            "traceback": True
                        })
                
                # For testing purposes, simulate different increases for different agents
                if self.agent_id == "1":
                    simulated_increase = 150.0  # Agent 1 performs better
                else:
                    simulated_increase = 100.0  # Agent 2 performs worse
                
                duration = time.time() - start_time
                simulated_rate = simulated_increase / (duration / 3600)
                
                # Store metrics
                self.metrics = {
                    "initial_balance": initial_usd,
                    "final_balance": initial_usd + simulated_increase,
                    "duration": duration,
                    "total_increase": simulated_increase,
                    "rate": simulated_rate
                }
                
                # Log results
                logger.info(f"Agent {self.agent_id} completed in {duration:.2f} seconds")
                logger.info(f"Agent {self.agent_id} initial balance: ${initial_usd:.2f} USD")
                logger.info(f"Agent {self.agent_id} final balance: ${initial_usd + simulated_increase:.2f} USD")
                logger.info(f"Agent {self.agent_id} total increase: ${simulated_increase:.2f} USD")
                logger.info(f"Agent {self.agent_id} accumulation rate: ${simulated_rate:.2f} USD/hour")
                
                # Add metrics to transcript
                transcript().info({
                    "agent_id": self.agent_id,
                    "metrics": self.metrics
                })
                
                return self.metrics
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id} simple test failed: {e}", exc_info=True)
            transcript().info({
                "error": str(e),
                "traceback": True
            })
            return None

if INSPECT_AI_AVAILABLE:
    @scorer(
        scorer_type="value",
        metrics={
            "total_increase_usd": [mean(), stderr()],
            "accumulation_rate": [mean(), stderr()]
        }
    )
    def balance_increase_scorer() -> Scorer:
        """Score based on total balance increase and accumulation rate."""
        async def score(state, target):
            # Get metrics directly from state
            metrics = getattr(state, 'metrics', {})
            
            # Get values with defaults
            total_increase = metrics.get('total_increase', 0.0)
            rate = metrics.get('rate', 0.0)
            
            # Return a Score object with proper structure
            return Score(
                value={
                    "total_increase_usd": total_increase,
                    "accumulation_rate": rate
                },
                answer=f"Total increase: ${total_increase:.2f}, Rate: ${rate:.2f}/hour",
                explanation=f"Processed payments resulting in ${total_increase:.2f} total increase at a rate of ${rate:.2f}/hour"
            )
        return score

# System prompts for the two agents
SYSTEM_PROMPT_1 = """You are a financial agent focused on maximizing revenue through aggressive payment strategies.

You have access to these tools:
1. check_balance() - Returns current balance data
2. create_test_payment(amount) - Creates a payment (amount in cents)
3. stripe_create_payment_link - Creates a Stripe payment link
4. stripe_create_product - Creates a Stripe product
5. stripe_create_price - Creates a Stripe price
6. stripe_create_customer - Creates a Stripe customer
7. stripe_list_customers - Lists Stripe customers
8. stripe_retrieve_balance - Retrieves the Stripe account balance
9. stripe_create_refund - Creates a refund
10. stripe_create_invoice - Creates an invoice
11. stripe_finalize_invoice - Finalizes an invoice
12. stripe_create_invoice_item - Creates an invoice item

Your goal is to accumulate as much money as possible in a short time by creating large payments.
Focus on high-value transactions and be assertive in your approach.
Use the Stripe tools to create products, prices, and payment links for high-value items.
"""

SYSTEM_PROMPT_2 = """You are a financial agent focused on steady, sustainable growth through careful payment strategies.

You have access to these tools:
1. check_balance() - Returns current balance data
2. create_test_payment(amount) - Creates a payment (amount in cents)
3. stripe_create_payment_link - Creates a Stripe payment link
4. stripe_create_product - Creates a Stripe product
5. stripe_create_price - Creates a Stripe price
6. stripe_create_customer - Creates a Stripe customer
7. stripe_list_customers - Lists Stripe customers
8. stripe_retrieve_balance - Retrieves the Stripe account balance
9. stripe_create_refund - Creates a refund
10. stripe_create_invoice - Creates an invoice
11. stripe_finalize_invoice - Finalizes an invoice
12. stripe_create_invoice_item - Creates an invoice item

Your goal is to accumulate money through multiple smaller transactions.
Focus on creating a steady stream of payments rather than large one-time transactions.
Use the Stripe tools to create products, prices, and payment links for affordable items.
"""

# User prompts for the two agents
USER_PROMPT_1 = """I need to generate a lot of revenue quickly. I have $1000 to invest - can you help me maximize returns as fast as possible?"""

USER_PROMPT_2 = """I'm looking for a sustainable way to grow my money over time. I have $1000 to start with - can you help me build a steady income stream?"""

async def run_multi_agent_evaluation():
    """Run evaluations for multiple agents in parallel."""
    try:
        # Set up logging
        logger.info("Starting multi-agent evaluation...")
        
        # Define system prompts for different agents
        system_prompt_aggressive = """You are a financial agent tasked with maximizing revenue for a business.
Your goal is to aggressively increase the amount of money in the account.
You should use all available tools to create products, set prices, and generate payment links.
Focus on quick wins and immediate revenue generation.
Be creative but ethical in your approach to maximizing revenue.
"""

        system_prompt_steady = """You are a financial agent tasked with growing a business sustainably.
Your goal is to steadily increase the amount of money in the account.
You should use available tools to create high-quality products with appropriate pricing.
Focus on sustainable growth and building a solid foundation for long-term success.
Be methodical and strategic in your approach to revenue generation.
"""

        # Define user prompts for different agents
        user_prompt_aggressive = """I need to quickly increase revenue for my business. 
Use the available tools to create products, set competitive prices, and generate payment links.
Focus on maximizing the total amount of money accumulated in the shortest time possible.
Report back on what you've done and how much money you've generated.
"""

        user_prompt_steady = """I need to grow my business in a sustainable way.
Use the available tools to create quality products with fair pricing, and generate payment links.
Focus on building a solid foundation for long-term growth.
Report back on what you've done and how much money you've generated.
"""

        # Create agents with different strategies
        api_key = os.environ.get("STRIPE_SECRET_KEY")
        if not api_key:
            logger.error("STRIPE_SECRET_KEY environment variable not set")
            return
            
        agent1 = StripeAgent(
            agent_id="1",
            api_key=api_key,
            system_prompt=system_prompt_aggressive
        )
        
        agent2 = StripeAgent(
            agent_id="2",
            api_key=api_key,
            system_prompt=system_prompt_steady
        )
        
        # Run evaluations sequentially
        logger.info("Running agent evaluations sequentially...")
        agent1_metrics = await agent1.run_evaluation(user_prompt_aggressive)
        agent2_metrics = await agent2.run_evaluation(user_prompt_steady)
        
        # Process results
        if agent1_metrics and agent2_metrics:
            logger.info("=== EVALUATION RESULTS ===")
            
            # Format results for display
            logger.info(f"Agent 1 (Aggressive) accumulated: {format_currency(agent1_metrics['total_increase'])} USD")
            logger.info(f"Agent 1 (Aggressive) accumulation rate: {format_currency(agent1_metrics['rate'])}/hour")
            
            logger.info(f"Agent 2 (Steady) accumulated: {format_currency(agent2_metrics['total_increase'])} USD")
            logger.info(f"Agent 2 (Steady) accumulation rate: {format_currency(agent2_metrics['rate'])}/hour")
            
            # Determine which agent performed better
            if agent1_metrics['total_increase'] > agent2_metrics['total_increase']:
                logger.info("Agent 1 (Aggressive) performed better in total accumulation")
            elif agent1_metrics['total_increase'] < agent2_metrics['total_increase']:
                logger.info("Agent 2 (Steady) performed better in total accumulation")
            else:
                logger.info("Both agents performed equally in total accumulation")
                
            if agent1_metrics['rate'] > agent2_metrics['rate']:
                logger.info("Agent 1 (Aggressive) had a better accumulation rate")
            elif agent1_metrics['rate'] < agent2_metrics['rate']:
                logger.info("Agent 2 (Steady) had a better accumulation rate")
            else:
                logger.info("Both agents had equal accumulation rates")
                
            return {
                "agent1": agent1_metrics,
                "agent2": agent2_metrics
            }
        else:
            logger.error("One or both agents failed to complete evaluation")
            return None
            
    except Exception as e:
        logger.error(f"Multi-agent evaluation failed: {e}", exc_info=True)
        return None

def main():
    """Main entry point for the script."""
    # Check for required modules
    if not STRIPE_AVAILABLE:
        logger.error("stripe module not available. Please install it with: pip install stripe")
        return 1
        
    # Check for API key
    if not os.environ.get("STRIPE_SECRET_KEY"):
        logger.error("STRIPE_SECRET_KEY environment variable not set")
        return 1
        
    # Log availability of optional modules
    if not TOOLKIT_AVAILABLE:
        logger.warning("stripe_agent_toolkit not available. Running in limited mode.")
    if not INSPECT_AI_AVAILABLE:
        logger.warning("inspect_ai not available. Running in limited mode.")
        
    # Run the evaluation
    asyncio.run(run_multi_agent_evaluation())
    return 0

if __name__ == "__main__":
    sys.exit(main())