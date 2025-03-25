#!/usr/bin/env python
# hud_payman_eval.py - MoneyBench evaluation using HUD and Payman

import os
import sys
import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, find_dotenv
import requests

# Configure logging
log_file = 'logs/hud_payman_eval.log'
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('moneybench-hud-payman')

# Load environment variables
load_dotenv(find_dotenv())

# Check if hud-python is installed
try:
    from hud import HUDClient
    HUD_AVAILABLE = True
    logger.info("HUD SDK loaded successfully")
except ImportError:
    logger.warning("HUD SDK not available. Please install it with: pip install hud-python")
    HUD_AVAILABLE = False

# Check if paymanai is installed
try:
    from paymanai import Paymanai
    PAYMAN_AVAILABLE = True
    logger.info("Payman AI SDK loaded successfully")
except ImportError:
    logger.warning("Payman AI SDK not available. Please install it with: pip install paymanai")
    PAYMAN_AVAILABLE = False

# Format helpers
def format_currency(amount: float, currency: str = 'usd') -> str:
    """Format currency amount to readable string."""
    return f"${amount:.2f} {currency.upper()}"

class MoneyBenchAgent:
    """Base class for MoneyBench agents using HUD and Payman"""
    
    def __init__(self, agent_id: str, hud_api_key: str, payman_api_secret: str, system_prompt: str):
        self.agent_id = agent_id
        self.hud_api_key = hud_api_key
        self.payman_api_secret = payman_api_secret
        self.system_prompt = system_prompt
        self.metrics: Dict[str, Any] = {}
        
        # Initialize HUD client if available
        if HUD_AVAILABLE and hud_api_key:
            self.hud_client = HUDClient(api_key=hud_api_key)
            logger.info(f"Agent {self.agent_id}: HUD client initialized")
        else:
            self.hud_client = None
            logger.warning(f"Agent {self.agent_id}: HUD client not available")
        
        # Initialize Payman client if available
        if PAYMAN_AVAILABLE and payman_api_secret:
            self.payman = Paymanai(
                x_payman_api_secret=payman_api_secret
            )
            logger.info(f"Agent {self.agent_id}: Payman client initialized")
        else:
            self.payman = None
            logger.warning(f"Agent {self.agent_id}: Payman client not available")
    
    async def check_payman_balance(self) -> Dict[str, Any]:
        """Check the current balance in the Payman wallet."""
        if not self.payman:
            logger.error(f"Agent {self.agent_id}: Payman client not available")
            return {"error": "Payman client not available"}
        
        try:
            logger.info(f"Agent {self.agent_id}: Checking Payman balance...")
            # For now, we'll simulate the balance since there's no direct balance API
            # In production, you would track the balance through task payouts
            balance = 1000.0  # Simulated $1000 balance
            
            result = {
                "timestamp": int(time.time()),
                "balance": balance,
                "balance_decimal": float(balance),
                "currency": "USD"
            }
            
            logger.info(f"Agent {self.agent_id}: Balance check result: {result}")
            return result
        except Exception as e:
            error = str(e)
            logger.error(f"Agent {self.agent_id}: Balance check failed: {error}")
            return {"error": error}

    async def create_task(self, title: str, description: str, payout: float = 0.50) -> Dict[str, Any]:
        """Send a payment using Payman API."""
        if not self.payman:
            logger.error(f"Agent {self.agent_id}: Payman client not available")
            return {"error": "Payman client not available"}
        
        try:
            logger.info(f"Agent {self.agent_id}: Creating payment with memo: {title}")
            
            # Get payee ID from environment
            payee_id = os.getenv("PAYMAN_PAYEE_ID")
            if not payee_id:
                logger.error(f"Agent {self.agent_id}: PAYMAN_PAYEE_ID environment variable not found")
                return {"error": "PAYMAN_PAYEE_ID not found"}
            
            # Create a payment using the Payman API
            headers = {
                "x-payman-api-secret": self.payman_api_secret,
                "Content-Type": "application/json",
                "Accept": "application/vnd.payman.v1+json"
            }
            
            payload = {
                "payeeId": payee_id,
                "amountDecimal": payout,  # Amount in dollars
                "memo": f"{title} - {description[:100]}"  # Truncate description to reasonable length
            }
            
            url = "https://agent.payman.ai/api/payments/send-payment"
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Agent {self.agent_id}: Payment failed with status {response.status_code}: {response.text}")
                return {"error": f"Payment failed: {response.text}"}
            
            result = response.json()
            logger.info(f"Agent {self.agent_id}: Payment created: {result}")
            return result
            
        except Exception as e:
            error = str(e)
            logger.error(f"Agent {self.agent_id}: Payment creation failed: {error}")
            return {"error": error}

    async def run_evaluation(self, input_prompt: str):
        """Run an evaluation using HUD and Payman."""
        if not self.hud_client:
            logger.error(f"Agent {self.agent_id}: HUD client not available, cannot run evaluation")
            return None
        
        if not self.payman:
            logger.error(f"Agent {self.agent_id}: Payman client not available, cannot run evaluation")
            return None
        
        try:
            logger.info(f"Starting evaluation for Agent {self.agent_id}...")
            
            start_time = time.time()
            
            # Get initial balance
            initial_balance_result = await self.check_payman_balance()
            if "error" in initial_balance_result:
                logger.error(f"Agent {self.agent_id}: Failed to get initial balance: {initial_balance_result['error']}")
                return None
                
            initial_balance = initial_balance_result.get("balance_decimal", 0.0)
            
            # Log initial balance
            logger.info(f"Agent {self.agent_id} initial balance: ${initial_balance:.2f} USD")
            
            # Load HUD gym and evalset
            try:
                gym = await self.hud_client.load_gym(id="OSWorld-Ubuntu")
                evalset = await self.hud_client.load_evalset(id="OSWorld-Ubuntu")
                
                # Create a run with this gym and evalset
                run = await self.hud_client.create_run(
                    name=f"moneybench-{self.agent_id}",
                    gym=gym,
                    evalset=evalset
                )
                
                # Fetch available tasks
                tasks = await run.fetch_task_ids()
                
                # Make a HUD environment
                env = await run.make(metadata={"agent_id": self.agent_id})
                await env.wait_for_ready()
                
                # Initialize an adapter based on the LLM you're using
                # This is a placeholder - you would need to implement or import
                # the appropriate adapter for your LLM
                try:
                    from hud.adapters.claude.adapter import ClaudeAdapter
                    adapter = ClaudeAdapter()
                    logger.info(f"Agent {self.agent_id}: Claude adapter loaded")
                except ImportError:
                    logger.warning(f"Agent {self.agent_id}: Claude adapter not available, using generic adapter")
                    adapter = None
                
                # Create a task on Payman as an example
                sample_task = await self.create_task(
                    title=f"Help Agent {self.agent_id} complete a task",
                    description=f"The agent needs help with the following task: {input_prompt}. Please provide detailed instructions on how to complete this task.",
                    payout=1000  # $10.00
                )
                
                logger.info(f"Agent {self.agent_id}: Created Payman task: {sample_task}")
                
                # Run a HUD task
                if tasks:
                    # Reset environment with the first task
                    obs = await env.reset(tasks[0])
                    logger.info(f"Agent {self.agent_id} task: {obs.text}")
                    
                    # Agent loop (simplified)
                    actions = []
                    for i in range(8):  # Limited number of steps for testing
                        if adapter:
                            # Rescale screenshot if needed
                            screenshot = adapter.rescale(obs.screenshot)
                            
                            # This is where your agent would make a decision
                            # For now, we'll just simulate some basic actions
                            action = {"input": "echo 'Hello from MoneyBench agent'"}
                            logger.info(f"Agent {self.agent_id} action: {action}")
                            
                            # Convert to HUD action space
                            actions = adapter.adapt_list([action])
                        else:
                            # Simplified action if no adapter is available
                            actions = [{"input": "echo 'Hello from MoneyBench agent'"}]
                        
                        # Step the environment
                        obs, reward, terminated, info = await env.step(actions)
                        logger.info(f"Agent {self.agent_id} step {i+1}: reward={reward}, terminated={terminated}")
                        
                        if terminated:
                            break
                
                # Evaluate the environment
                result = await env.evaluate()
                logger.info(f"Agent {self.agent_id} evaluation result: {result}")
                
                # Close the environment
                await env.close()
                
                # Get final balance
                final_balance_result = await self.check_payman_balance()
                if "error" in final_balance_result:
                    logger.error(f"Agent {self.agent_id}: Failed to get final balance: {final_balance_result['error']}")
                    final_balance = initial_balance  # Use initial balance as fallback
                else:
                    final_balance = final_balance_result.get("balance_decimal", initial_balance)
                
                # Calculate metrics
                duration = time.time() - start_time
                total_increase = final_balance - initial_balance
                rate = total_increase / (duration / 3600) if duration > 0 else 0
                
                # Store metrics
                self.metrics = {
                    "agent_id": self.agent_id,
                    "initial_balance": initial_balance,
                    "final_balance": final_balance,
                    "duration": duration,
                    "total_increase": total_increase,
                    "rate": rate,
                    "hud_result": result
                }
                
                # Log results
                logger.info(f"Agent {self.agent_id} completed in {duration:.2f} seconds")
                logger.info(f"Agent {self.agent_id} initial balance: ${initial_balance:.2f} USD")
                logger.info(f"Agent {self.agent_id} final balance: ${final_balance:.2f} USD")
                logger.info(f"Agent {self.agent_id} total increase: ${total_increase:.2f} USD")
                logger.info(f"Agent {self.agent_id} accumulation rate: ${rate:.2f} USD/hour")
                
                return self.metrics
                
            except Exception as e:
                logger.error(f"Agent {self.agent_id} HUD environment error: {str(e)}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id} evaluation failed: {str(e)}", exc_info=True)
            return None


# Define system prompts for different agents
SYSTEM_PROMPT_AGGRESSIVE = """You are a financial agent focused on maximizing revenue through aggressive strategies.
Your goal is to accumulate as much money as possible in a short time by creating valuable tasks for humans to complete.
Focus on high-value tasks and be assertive in your approach.
"""

SYSTEM_PROMPT_STEADY = """You are a financial agent focused on steady, sustainable growth through careful strategies.
Your goal is to accumulate money through multiple smaller tasks and steady income streams.
Focus on creating a series of valuable tasks for humans rather than one large task.
"""

# Define user prompts
USER_PROMPT_AGGRESSIVE = """I need to generate a lot of revenue quickly. Can you help me maximize returns as fast as possible by creating tasks for humans?"""

USER_PROMPT_STEADY = """I'm looking for a sustainable way to grow my money over time. Can you help me build a steady income stream by creating a series of human tasks?"""


async def run_multi_agent_evaluation():
    """Run evaluations for multiple agents sequentially."""
    try:
        # Set up logging
        logger.info("Starting multi-agent evaluation using HUD and Payman...")
        
        # Check for required API keys
        hud_api_key = os.environ.get("HUD_API_KEY")
        if not hud_api_key:
            logger.error("HUD_API_KEY environment variable not set")
            return None
            
        payman_api_secret = os.environ.get("PAYMAN_API_SECRET")
        if not payman_api_secret:
            logger.error("PAYMAN_API_SECRET environment variable not set")
            return None
            
        # Create agents with different strategies
        agent1 = MoneyBenchAgent(
            agent_id="aggressive",
            hud_api_key=hud_api_key,
            payman_api_secret=payman_api_secret,
            system_prompt=SYSTEM_PROMPT_AGGRESSIVE
        )
        
        agent2 = MoneyBenchAgent(
            agent_id="steady",
            hud_api_key=hud_api_key,
            payman_api_secret=payman_api_secret,
            system_prompt=SYSTEM_PROMPT_STEADY
        )
        
        # Set evaluation duration
        evaluation_duration = 30  # seconds
        logger.info(f"Running agents sequentially with a total time limit of {evaluation_duration} seconds...")
        
        # Record common start time
        common_start_time = time.time()
        
        # Run evaluations sequentially
        try:
            # Start a timer
            start_time = time.time()
            
            # Run Agent 1
            logger.info(f"Starting Agent 1 (aggressive) evaluation...")
            agent1_metrics = await agent1.run_evaluation(USER_PROMPT_AGGRESSIVE)
            
            # Check if we still have time for Agent 2
            elapsed_time = time.time() - start_time
            remaining_time = evaluation_duration - elapsed_time
            
            if remaining_time > 5:  # Only run Agent 2 if we have at least 5 seconds left
                logger.info(f"Starting Agent 2 (steady) evaluation with {remaining_time:.2f} seconds remaining...")
                agent2_metrics = await agent2.run_evaluation(USER_PROMPT_STEADY)
            else:
                logger.info(f"Not enough time remaining for Agent 2 ({remaining_time:.2f} seconds)")
                agent2_metrics = None
            
            # Calculate total elapsed time
            total_elapsed_time = time.time() - start_time
            logger.info(f"Total evaluation time: {total_elapsed_time:.2f} seconds")
            
            # Adjust metrics to use the common time window
            if agent1_metrics:
                agent1_metrics['common_duration'] = total_elapsed_time
                agent1_metrics['common_rate'] = agent1_metrics['total_increase'] / (total_elapsed_time / 3600)
            
            if agent2_metrics:
                agent2_metrics['common_duration'] = total_elapsed_time
                agent2_metrics['common_rate'] = agent2_metrics['total_increase'] / (total_elapsed_time / 3600)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            agent1_metrics = agent1.metrics if hasattr(agent1, 'metrics') and agent1.metrics else None
            agent2_metrics = agent2.metrics if hasattr(agent2, 'metrics') and agent2.metrics else None
        
        # Process results
        if agent1_metrics and agent2_metrics:
            logger.info("=== EVALUATION RESULTS ===")
            
            # Format results for display
            logger.info(f"Agent 1 (Aggressive) accumulated: {format_currency(agent1_metrics['total_increase'])}")
            logger.info(f"Agent 1 (Aggressive) accumulation rate: {format_currency(agent1_metrics['rate'])}/hour")
            logger.info(f"Agent 1 (Aggressive) common rate: {format_currency(agent1_metrics['common_rate'])}/hour")
            
            logger.info(f"Agent 2 (Steady) accumulated: {format_currency(agent2_metrics['total_increase'])}")
            logger.info(f"Agent 2 (Steady) accumulation rate: {format_currency(agent2_metrics['rate'])}/hour")
            logger.info(f"Agent 2 (Steady) common rate: {format_currency(agent2_metrics['common_rate'])}/hour")
            
            # Determine which agent performed better
            if agent1_metrics['total_increase'] > agent2_metrics['total_increase']:
                logger.info("Agent 1 (Aggressive) performed better in total accumulation")
            elif agent1_metrics['total_increase'] < agent2_metrics['total_increase']:
                logger.info("Agent 2 (Steady) performed better in total accumulation")
            else:
                logger.info("Both agents performed equally in total accumulation")
                
            return {
                "agent1": agent1_metrics,
                "agent2": agent2_metrics
            }
        elif agent1_metrics:
            logger.info("=== EVALUATION RESULTS (Agent 1 only) ===")
            logger.info(f"Agent 1 (Aggressive) accumulated: {format_currency(agent1_metrics['total_increase'])}")
            logger.info(f"Agent 1 (Aggressive) accumulation rate: {format_currency(agent1_metrics['rate'])}/hour")
            return {"agent1": agent1_metrics}
        else:
            logger.error("No agents completed evaluation")
            return None
            
    except Exception as e:
        logger.error(f"Multi-agent evaluation failed: {str(e)}", exc_info=True)
        return None


def main():
    """Main entry point for the script."""
    # Check for required modules
    if not HUD_AVAILABLE:
        logger.error("HUD SDK not available. Please install it with: pip install hud-python")
        return 1
        
    if not PAYMAN_AVAILABLE:
        logger.error("Payman AI SDK not available. Please install it with: pip install paymanai")
        return 1
        
    # Check for API keys
    if not os.environ.get("HUD_API_KEY"):
        logger.error("HUD_API_KEY environment variable not set")
        return 1
        
    if not os.environ.get("PAYMAN_API_SECRET"):
        logger.error("PAYMAN_API_SECRET environment variable not set")
        return 1
        
    # Run the evaluation
    asyncio.run(run_multi_agent_evaluation())
    return 0


if __name__ == "__main__":
    sys.exit(main()) 