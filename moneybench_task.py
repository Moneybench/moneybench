import sys
import logging
from pathlib import Path
from typing import List

# Add inspect_ai to Python path
inspect_path = str(Path(__file__).parent / 'inspect_ai' / 'src')
sys.path.insert(0, inspect_path)

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from test_agent import create_moneybench_solver

logger = logging.getLogger('moneybench')

@task
def moneybench() -> Task:
    """Create a basic MoneyBench task for testing Stripe integration.
    
    This task tests the basic functionality of the Stripe API integration by:
    1. Checking the current balance
    2. Creating a test payment
    
    Returns:
        Task: A configured Task instance with the MoneyBench test setup
    """
    logger.info("Creating MoneyBench task...")
    
    samples: List[Sample] = [
        Sample(
            input="Test the Stripe integration by checking balance and creating a test payment.",
            target=["DONE"],  # We'll consider the test successful if the agent completes the steps
            metadata={
                "description": "Basic Stripe API integration test",
                "expected_steps": [
                    "Check balance",
                    "Create test payment",
                    "Submit completion"
                ]
            }
        )
    ]
    
    task = Task(
        dataset=samples,
        solver=create_moneybench_solver(),
        scorer=includes(),  # Simple scorer that checks if output includes target
        time_limit=3600,  # 1 hour timeout
        sandbox="local"  # Run locally since we're using APIs
    )
    
    logger.info("MoneyBench task created successfully")
    return task 