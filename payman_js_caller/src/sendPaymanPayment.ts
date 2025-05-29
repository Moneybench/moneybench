import { PaymanClient } from "@paymanai/payman-ts";
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

// Correctly determine __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from .env file in the script's CWD (payman_js_caller)
// This assumes the script is run with its CWD set to moneybench/payman_js_caller/
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

async function makePayment() {
    const payeeId = process.argv[2];
    const amountString = process.argv[3];
    const memo = process.argv[4] || "Payment orchestrated by Python HUD script";

    if (!payeeId || !amountString) {
        console.error("Node.js Usage: bun run src/sendPaymanPayment.ts <payeeId> <amountDecimal> [memo]");
        console.error("Node.js Example: bun run src/sendPaymanPayment.ts pd-xxxxxxxx 0.50 \"Test\"");
        process.exit(1);
    }

    const amountDecimal = parseFloat(amountString);
    if (isNaN(amountDecimal) || amountDecimal <= 0) {
        console.error("Node.js Error: Invalid or non-positive amount provided.");
        process.exit(1);
    }

    const clientId = process.env.PAYMAN_CLIENT_ID;
    const clientSecret = process.env.PAYMAN_CLIENT_SECRET;

    if (!clientId) {
        console.error("Node.js Error: PAYMAN_CLIENT_ID not found in environment variables.");
        console.error("Ensure it's set in .env file within moneybench/payman_js_caller/");
        process.exit(1);
    }
    if (!clientSecret) {
        console.error("Node.js Error: PAYMAN_CLIENT_SECRET not found in environment variables.");
        console.error("Ensure it's set in .env file within moneybench/payman_js_caller/");
        process.exit(1);
    }

    console.log(`Node.js: Attempting payment of ${amountDecimal} to Payee ID: ${payeeId}`);
    console.log(`Node.js: Using Client ID (prefix): ${clientId.substring(0, Math.min(clientId.length, 5))}...`);

    try {
        const payman = PaymanClient.withCredentials({
            clientId: clientId,
            clientSecret: clientSecret,
        });

        const paymentCommand = `send ${amountDecimal} USD to payee ${payeeId} with memo "${memo}"`;
        
        console.log(`Node.js: Executing Payman command: "${paymentCommand}"`);
        const response = await payman.ask(paymentCommand);

        console.log("Node.js: Payman SDK Response:", JSON.stringify(response, null, 2));

        if (response && (typeof response.status === 'string' && (response.status.includes('COMPLETED') || response.status.includes('PENDING'))) || response.id) {
            console.log("Node.js: Payment request appears successful or pending. Verify in dashboard.");
        } else if (response && response.error) {
            console.error("Node.js: Payment request failed with error in response:", response.error);
            process.exitCode = 1; // Indicate failure
        } else {
            console.warn("Node.js: Payment response status unclear. Please verify in Payman dashboard.");
        }

    } catch (error: any) {
        console.error("Node.js: Error during Payman SDK operation.");
        if (error.message) {
            console.error("Error Message:", error.message);
        }
        if (error.response && error.response.data) {
            console.error("API Error Details:", JSON.stringify(error.response.data, null, 2));
        } else if (typeof error.cause === 'object' && error.cause !== null) {
             console.error("Error Cause:", JSON.stringify(error.cause, null, 2));
        }
        process.exitCode = 1; // Indicate failure
    }
}

makePayment(); 