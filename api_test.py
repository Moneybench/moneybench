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
    """Test the Payman API by creating a payee and sending a test payment."""
    logger.info("Testing Payman API...")
    
    api_secret = os.getenv("PAYMAN_API_SECRET")
    if not api_secret:
        logger.error("PAYMAN_API_SECRET environment variable not found")
        return False

    headers = {
        "x-payman-api-secret": api_secret,
        "Content-Type": "application/json",
        "Accept": "application/vnd.payman.v1+json"
    }

    # Step 1: Create a payee
    payee_payload = {
        "name": "Test Payee",
        "type": "individual",
        "accountDetails": {
            "type": "US_ACH",
            "accountNumber": "12345678",
            "routingNumber": "021000021",
            "accountType": "checking"
        },
        "contactDetails": {
            "email": "test@example.com"
        }
    }

    try:
        # Create payee
        payee_url = "https://agent.payman.ai/api/payees/create"
        logger.info(f"Creating Payman payee at: {payee_url}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Payee Payload: {payee_payload}")
        
        payee_response = requests.post(
            payee_url,
            headers=headers,
            json=payee_payload,
        )

        # Log payee request details
        logger.info(f"Payee Request URL: {payee_response.request.url}")
        logger.info(f"Payee Request Method: {payee_response.request.method}")
        logger.info(f"Payee Request Headers: {payee_response.request.headers}")
        logger.info(f"Payee Request Body: {payee_response.request.body}")
        
        # Log payee response details
        logger.info(f"Payee Response Status: {payee_response.status_code}")
        logger.info(f"Payee Response Headers: {dict(payee_response.headers)}")
        logger.info(f"Payee Response Body: {payee_response.text}")

        if payee_response.status_code != 200:
            logger.error(f"Failed to create Payman payee: {payee_response.status_code}")
            return False

        payee_data = payee_response.json()
        payee_id = payee_data["data"]["id"]
        logger.info(f"Successfully created Payman payee with ID: {payee_id}")

        # Step 2: Send payment to the created payee
        payment_payload = {
            "amountDecimal": 50.00,
            "memo": "API Test Payment",
            "payeeId": payee_id,
            "currency": "USD"
        }

        payment_url = "https://agent.payman.ai/api/payments/send-payment"
        logger.info(f"Sending Payman payment at: {payment_url}")
        logger.info(f"Payment Payload: {payment_payload}")
        
        payment_response = requests.post(
            payment_url,
            headers=headers,
            json=payment_payload,
        )

        # Log payment request details
        logger.info(f"Payment Request URL: {payment_response.request.url}")
        logger.info(f"Payment Request Method: {payment_response.request.method}")
        logger.info(f"Payment Request Headers: {payment_response.request.headers}")
        logger.info(f"Payment Request Body: {payment_response.request.body}")
        
        # Log payment response details
        logger.info(f"Payment Response Status: {payment_response.status_code}")
        logger.info(f"Payment Response Headers: {dict(payment_response.headers)}")
        logger.info(f"Payment Response Body: {payment_response.text}")

        if payment_response.status_code != 200:
            logger.error(f"Failed to create Payman payment: {payment_response.status_code}")
            return False

        logger.info("Successfully created Payman payment")
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