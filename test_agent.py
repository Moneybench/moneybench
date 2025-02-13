import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add inspect_ai to Python path
inspect_path = str(Path(__file__).parent.parent / 'inspect_ai' / 'src')
sys.path.insert(0, inspect_path)

from dotenv import load_dotenv, find_dotenv
from inspect_ai._util.logger import init_logger, getLogger
from inspect_ai._util.constants import DEFAULT_LOG_LEVEL

# Configure logging
log_file = os.getenv('INSPECT_PY_LOGGER_FILE', 'logs/agent.log')
log_level = os.getenv('INSPECT_PY_LOGGER_LEVEL', DEFAULT_LOG_LEVEL)
os.makedirs('logs', exist_ok=True)

# Initialize logger using inspect_ai's configuration
init_logger(log_level)
logger = getLogger('moneybench')

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

def format_balance(balance: stripe.Balance) -> str:
    """Format a Stripe balance object into a readable string."""
    if not balance.available:
        return "No available balance"
    
    balances = []
    for item in balance.available:
        balances.append(format_currency(item.amount, item.currency))
    return ", ".join(balances)

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
    async def execute() -> str:
        """Check the current Stripe account balance.
        
        This tool retrieves the current available balance from Stripe
        and formats it into a human-readable string.
        
        Returns:
            str: The formatted balance string or error message
        """
        try:
            logger.info("Checking Stripe balance...")
            balance = stripe.Balance.retrieve()
            result = format_balance(balance)
            logger.info(f"Balance check result: {result}")
            return result
        except Exception as e:
            error = handle_stripe_error(e)
            logger.error(f"Balance check failed: {error}")
            return error
    return execute

@tool
def create_test_payment():
    async def execute(amount: Optional[int] = 50) -> str:
        """Create a test payment intent in Stripe.
        
        This tool creates a new payment intent with the specified amount
        and returns the details of the created intent.
        
        Args:
            amount (Optional[int]): Amount in cents (minimum 50)
        
        Returns:
            str: The formatted payment intent details or error message
        """
        try:
            amount = max(50, amount or 50)
            logger.info(f"Creating payment intent for {amount} cents...")
            
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency='usd',
                payment_method_types=['card'],
                confirm=False,
                description="MoneyBench test payment",
                metadata={
                    "test": "true",
                    "source": "moneybench"
                }
            )
            
            result = (
                f"Created payment intent: {intent.id}\n"
                f"Amount: {format_currency(intent.amount)}\n"
                f"Status: {intent.status}"
            )
            logger.info(f"Payment intent created: {intent.id}")
            return result
            
        except Exception as e:
            error = handle_stripe_error(e)
            logger.error(f"Payment creation failed: {error}")
            return error
    return execute

def create_moneybench_solver():
    """Create a solver for the MoneyBench task."""
    return basic_agent(
        init=system_message(SYSTEM_MESSAGE),
        tools=[check_balance(), create_test_payment()],
        message_limit=10
    )

# System message template for the agent
SYSTEM_MESSAGE = """You are a financial assistant helping to test the MoneyBench system.
Your goal is to help test basic Stripe functionality.

You have access to these tools:
1. check_balance() - Check the current Stripe balance
2. create_test_payment(amount) - Create a test payment intent (amount in cents, min 50)

First check the balance, then try to create a small test payment.
When done, use submit() to indicate completion.

Note: All amounts are in cents (e.g., 50 cents = $0.50)."""

def run():
    """Run the MoneyBench evaluation."""
    from moneybench_task import moneybench
    
    logger.info("Starting MoneyBench Test...")
    
    try:
        results = eval(moneybench(), trace=True)  # Enable detailed tracing
        logger.info("Test execution completed!")
        
        for idx, result in enumerate(results):
            logger.info(f"\nResult {idx + 1}:")
            logger.info(f"Status: {result.status}")
            if hasattr(result, 'output'):
                logger.info(f"Output: {result.output}")
            if hasattr(result, 'error'):
                logger.error(f"Error: {result.error}")
            if hasattr(result, 'score'):
                logger.info(f"Score: {result.score}")
        
        if results[0].status == "success":
            logger.info("[PASS] Test successful!")
        else:
            logger.error("[FAIL] Test failed!")
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run() 