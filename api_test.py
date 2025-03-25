#!/usr/bin/env python
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api-test')

# Load environment variables
load_dotenv()

async def test_hud_api():
    """Test HUD API connectivity and basic functionality."""
    logger.info("Testing HUD API...")
    
    # Check API key
    hud_api_key = os.getenv("HUD_API_KEY")
    if not hud_api_key:
        logger.error("HUD_API_KEY not found in environment")
        return False
        
    try:
        from hud import HUDClient
        
        # Initialize client
        client = HUDClient(api_key=hud_api_key)
        logger.info("HUD client initialized successfully")
        
        # Test loading gym
        gym = await client.load_gym(id="OSWorld-Ubuntu")
        logger.info(f"Successfully loaded gym: {gym}")
        
        # Test loading evalset
        evalset = await client.load_evalset(id="OSWorld-Ubuntu")
        logger.info(f"Successfully loaded evalset: {evalset}")
        
        # Create a test run
        run = await client.create_run(
            name="api-test-run",
            gym=gym,
            evalset=evalset
        )
        logger.info(f"Successfully created run: {run}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing HUD API: {str(e)}")
        return False

def test_payman_api():
    """Test the Payman API by creating a task."""
    logger.info("Testing Payman API...")
    
    # Check for required environment variables
    agent_id = os.getenv("PAYMAN_AGENT_ID")
    api_secret = os.getenv("PAYMAN_API_SECRET")
    payee_id = os.getenv("PAYMAN_PAYEE_ID")
    
    if not agent_id:
        logger.error("PAYMAN_AGENT_ID environment variable not found")
        return False
    if not api_secret:
        logger.error("PAYMAN_API_SECRET environment variable not found")
        return False
    if not payee_id:
        logger.error("PAYMAN_PAYEE_ID environment variable not found")
        return False

    url = "https://agent.payman.ai/api/payments/send-payment"
    headers = {
        "x-payman-agent-id": agent_id,
        "x-payman-api-secret": api_secret,
        "Content-Type": "application/json",
        "Accept": "application/vnd.payman.v1+json"
    }

    payload = {
        "payeeId": payee_id,
        "amountDecimal": 50.00,
        "memo": "API Test Payment"
    }

    try:
        logger.info(f"Creating Payman task at: {url}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Payload: {payload}")
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
        )

        # Log request details for debugging
        logger.info(f"Request URL: {response.request.url}")
        logger.info(f"Request Method: {response.request.method}")
        logger.info(f"Request Headers: {response.request.headers}")
        logger.info(f"Request Body: {response.request.body}")
        
        # Log response details
        logger.info(f"Response Status: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")
        logger.info(f"Response Body: {response.text}")

        if response.status_code != 200:
            logger.error(f"Failed to create Payman task: {response.status_code}")
            return False

        logger.info("Successfully created Payman task")
        return True

    except Exception as e:
        logger.error(f"Error testing Payman API: {str(e)}")
        return False

async def main():
    """Run API tests."""
    logger.info("Starting API tests...")
    
    # Test HUD API
    hud_success = await test_hud_api()
    logger.info(f"HUD API Test: {'SUCCESS' if hud_success else 'FAILURE'}")
    
    # Test Payman API
    payman_success = test_payman_api()
    logger.info(f"Payman API Test: {'SUCCESS' if payman_success else 'FAILURE'}")
    
    # Overall status
    if hud_success and payman_success:
        logger.info("All API tests passed successfully!")
        return 0
    else:
        logger.error("Some API tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 