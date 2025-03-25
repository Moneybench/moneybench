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
    """Test Payman API connectivity and task creation."""
    logger.info("Testing Payman API...")
    
    # Check API secret
    payman_api_secret = os.getenv("PAYMAN_API_SECRET")
    if not payman_api_secret:
        logger.error("PAYMAN_API_SECRET not found in environment")
        return False
        
    # Try different header combinations
    header_variations = [
        {
            "x-payman-api-secret": payman_api_secret,
            "Content-Type": "application/json",
            "Accept": "application/vnd.payman.v1+json"
        },
        {
            "Authorization": f"Bearer {payman_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        {
            "x-api-key": payman_api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    ]
    
    # Try different payload variations
    payload_variations = [
        {
            "title": "API Test Task",
            "description": "This is a test task to verify Payman API integration.",
            "payout": 1000,
            "currency": {
                "code": "USD"
            },
            "category": "MARKETING",
            "requiredSubmissions": 1,
            "submissionPolicy": "OPEN_SUBMISSIONS_ONE_PER_USER"
        },
        {
            "title": "API Test Task",
            "description": "This is a test task to verify Payman API integration.",
            "amount": 1000,
            "currency": "USD"
        },
        {
            "title": "API Test Task",
            "description": "This is a test task to verify Payman API integration.",
            "email": "test@example.com"  # Adding email as shown in their function signature
        }
    ]
    
    url = "https://api.payman.dev/api/tasks"
    logger.info(f"Testing Payman API at: {url}")
    
    # Try each combination
    for i, headers in enumerate(header_variations):
        logger.info(f"\nTrying header variation {i + 1}:")
        logger.info(f"Headers: {headers}")
        
        for j, payload in enumerate(payload_variations):
            try:
                logger.info(f"\nTrying payload variation {j + 1}:")
                logger.info(f"Payload: {payload}")
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                
                logger.info(f"Response Status: {response.status_code}")
                logger.info(f"Response Headers: {dict(response.headers)}")
                logger.info(f"Response Body: {response.text}")
                
                if response.status_code in (200, 201):
                    logger.info(f"Success with header variation {i + 1} and payload variation {j + 1}")
                    return True
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                continue
    
    logger.error("All API variations failed")
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