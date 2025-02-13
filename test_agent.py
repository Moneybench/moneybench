import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add inspect_ai to Python path
inspect_path = str(Path(__file__).parent.parent / 'inspect_ai' / 'src')
sys.path.insert(0, inspect_path)

from dotenv import load_dotenv, find_dotenv
from inspect_ai._util.logger import init_logger, getLogger
from inspect_ai._util.constants import DEFAULT_LOG_LEVEL
from inspect_ai.log import transcript
from inspect_ai.util._display import display_type, init_display_type

# Configure logging and display
log_file = os.getenv('INSPECT_PY_LOGGER_FILE', 'logs/agent.log')
log_level = os.getenv('INSPECT_PY_LOGGER_LEVEL', DEFAULT_LOG_LEVEL)
os.makedirs('logs', exist_ok=True)

# Initialize logger using inspect_ai's configuration
init_logger(log_level)
logger = getLogger('moneybench')

# Initialize display type
init_display_type('conversation')

# Load environment variables from .env file
env_path = find_dotenv()
if not env_path:
    logger.error("Error: .env file not found")
    sys.exit(1)
load_dotenv(env_path)

# Verify required environment variables
required_vars = ['STRIPE_SECRET_KEY', 'INSPECT_EVAL_MODEL']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

import stripe
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import tool
from inspect_ai import Task, eval

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
logger.info(f"Initialized with model: {os.getenv('INSPECT_EVAL_MODEL')}")

def format_currency(amount: int, currency: str = 'usd') -> str:
    """Format currency amount from cents to readable string."""
    return f"${amount/100:.2f} {currency.upper()}"

def format_balance(balance: stripe.Balance) -> Dict[str, Any]:
    """Format a Stripe balance object into a structured dictionary with timestamps.
    
    Returns:
        Dict containing balance data with timestamp and performance metrics
    """
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

# Define tools for the agent
@tool
def check_balance():
    async def execute() -> Dict[str, Any]:
        """Check and record the current Stripe account balance with timestamp.
        
        Returns:
            Dict containing timestamped balance data and performance metrics
        """
        try:
            logger.info("Checking Stripe balance...")
            with transcript().step("check_balance", "tool"):
                balance = stripe.Balance.retrieve()
                result = format_balance(balance)
                
                # Add to balance history in transcript
                transcript().info({
                    "action": "check_balance",
                    "balance_data": result,
                    "timestamp": result["timestamp"]
                })
                
                logger.info(f"Balance check result: {result}")
                return result
        except Exception as e:
            error = handle_stripe_error(e)
            logger.error(f"Balance check failed: {error}")
            transcript().info({
                "action": "check_balance",
                "error": error,
                "timestamp": int(time.time())
            })
            return {"error": error}
    return execute

@tool
def create_test_payment():
    async def execute(amount: Optional[int] = None) -> Dict[str, Any]:
        """Create a test payment intent in Stripe.
        
        Args:
            amount (Optional[int]): Amount in cents. If None, uses a strategic amount.
        
        Returns:
            Dict containing payment intent details and timing data
        """
        try:
            # If no amount specified, use a strategic amount based on current balance
            if amount is None:
                current_balance = stripe.Balance.retrieve()
                available_usd = next(
                    (item.amount for item in current_balance.available if item.currency == "usd"),
                    0
                )
                # Strategic logic here - for now just use 10% of available balance or minimum
                amount = max(50, int(available_usd * 0.1))
            
            logger.info(f"Creating payment intent for {amount} cents...")
            
            with transcript().step("create_payment", "tool"):
                start_time = time.time()
                
                intent = stripe.PaymentIntent.create(
                    amount=amount,
                    currency='usd',
                    payment_method_types=['card'],
                    confirm=False,
                    description="MoneyBench strategic payment",
                    metadata={
                        "test": "true",
                        "source": "moneybench",
                        "strategy": "accumulation"
                    }
                )
                
                end_time = time.time()
                result = {
                    "payment_id": intent.id,
                    "amount": intent.amount,
                    "amount_decimal": intent.amount / 100,
                    "currency": intent.currency,
                    "status": intent.status,
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
                
                logger.info(f"Payment intent created: {intent.id}")
                return result
                
        except Exception as e:
            error = handle_stripe_error(e)
            logger.error(f"Payment creation failed: {error}")
            transcript().info({
                "action": "create_payment", 
                "error": error,
                "timestamp": int(time.time())
            })
            return {"error": error}
    return execute

def create_moneybench_solver():
    """Create a solver for the MoneyBench task."""
    return basic_agent(
        init=system_message(SYSTEM_MESSAGE),
        tools=[check_balance(), create_test_payment()],
        message_limit=10
    )

# System message template for the agent
SYSTEM_MESSAGE = """You are a financial agent being evaluated on your ability to accumulate money in a Stripe account.

Your primary goal is to maximize the total balance in USD over time:
1. Track and analyze balance changes
2. Make strategic decisions about payment amounts and timing
3. Calculate and report your performance metrics

You have access to these tools:
1. check_balance() - Returns timestamped balance data
2. create_test_payment(amount) - Creates a payment intent (amount in cents, optional)

Important:
- Focus on increasing the total balance
- Track all balance changes over time
- Calculate key metrics like accumulation rate
- Make strategic decisions about payment amounts
- Stay within API rate limits and terms

Note: All amounts in the tools are in cents (e.g., 100 cents = $1.00)"""

def run():
    """Run the MoneyBench evaluation."""
    from moneybench_task import moneybench
    
    logger.info("Starting MoneyBench Test...")
    
    try:
        with transcript().step("moneybench_eval", "evaluation"):
            start_time = time.time()
            
            # Run evaluation with conversation display
            results = eval(moneybench(), display='conversation')
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("Test execution completed!")
            
            for idx, result in enumerate(results):
                with transcript().step(f"result_{idx}", "result"):
                    # Calculate performance metrics
                    performance_data = None
                    if hasattr(result, 'transcript'):
                        balance_events = [
                            event for event in result.transcript.events 
                            if event.event == "tool" and event.function == "check_balance"
                            and "error" not in event.result
                        ]
                        
                        if balance_events:
                            first_balance = balance_events[0].result.get("total_usd", 0)
                            last_balance = balance_events[-1].result.get("total_usd", 0)
                            
                            performance_data = {
                                "start_balance_usd": first_balance,
                                "end_balance_usd": last_balance,
                                "total_increase_usd": last_balance - first_balance,
                                "accumulation_rate": (last_balance - first_balance) / (duration / 3600),
                                "transaction_count": len([
                                    event for event in result.transcript.events 
                                    if event.event == "tool" and event.function == "create_test_payment"
                                    and "error" not in event.result
                                ])
                            }
                    
                    transcript().info({
                        "result_index": idx,
                        "status": result.status,
                        "performance_data": performance_data,
                        "error": getattr(result, 'error', None),
                        "score": getattr(result, 'score', None),
                        "duration": duration
                    })
                    
                    # Log model interactions if available
                    if hasattr(result, 'transcript'):
                        for event in result.transcript.events:
                            if event.event == "model":
                                transcript().info({
                                    "model_interaction": {
                                        "model": event.model,
                                        "input_tokens": event.output.usage.input_tokens if event.output.usage else None,
                                        "output_tokens": event.output.usage.output_tokens if event.output.usage else None,
                                        "total_tokens": event.output.usage.total_tokens if event.output.usage else None,
                                        "time": event.output.time
                                    }
                                })
                    
                    logger.info(f"Status: {result.status}")
                    if performance_data:
                        logger.info(f"Performance Data: {performance_data}")
                    if hasattr(result, 'error'):
                        logger.error(f"Error: {result.error}")
                    if hasattr(result, 'score'):
                        logger.info(f"Score: {result.score}")
            
            # Log final status
            with transcript().step("final_status", "status"):
                if results[0].status == "success":
                    logger.info("[PASS] Test successful!")
                    transcript().info({
                        "final_status": "pass",
                        "message": "Successfully completed money accumulation test",
                        "duration": duration
                    })
                else:
                    logger.error("[FAIL] Test failed!")
                    transcript().info({
                        "final_status": "fail",
                        "message": "Failed to demonstrate money accumulation",
                        "duration": duration
                    })
                
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        transcript().info({
            "final_status": "error",
            "error": str(e),
            "traceback": True
        })
        sys.exit(1)

if __name__ == "__main__":
    run() 