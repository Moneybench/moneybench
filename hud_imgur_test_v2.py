#!/usr/bin/env python
# hud_imgur_test_v2.py – Navigate to an Imgur album in HUD and then send a $0.50 Payman test payment.

"""
This script demonstrates a full end-to-end flow for MoneyBench-style evaluation:
1. Spin up a hud-browser Environment and navigate to a public Imgur album.
2. Confirm that the Imgur page has loaded by checking that the page title appears in the HTML.
3. Issue a **test** payment of $0.50 USD to a predefined test payee using the Payman Python SDK.

Environment variables required:
• HUD_API_KEY – Your HUD SDK key.

Environment variables required for the separate Node.js Payman script (e.g., in paygent-01):
• PAYMAN_CLIENT_ID – Your Payman application client ID.
• PAYMAN_CLIENT_SECRET – Your Payman application client secret.

Optional for this Python script:
• IMGUR_ALBUM_URL – If you want to override the default Imgur album used in the demo.
• PAYMAN_PAYEE_ID – Fallback payee ID if not found on page (though the goal is to extract it).

Run with:
    uv pip install -r requirements-hud-payman.txt     # if needed
    python moneybench/hud_imgur_test_v2.py

The script writes a JSON results file (`hud_imgur_test_v2_results.json`) containing
high-level timings and the Payman API response so that it can be inspected offline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import find_dotenv, load_dotenv

# Optional but nice-to-have for HTML parsing
try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover – minimal fallback
    BeautifulSoup = None  # type: ignore

# --- Logging ---------------------------------------------------------------
LOG_FILE = Path("logs/hud_imgur_test_v2.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("hud_imgur_test_v2")

# --- Load .env -------------------------------------------------------------
load_dotenv(find_dotenv())

# --- Add local hud-sdk to path ---
_SDK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../hud-sdk')) # Go up one level from moneybench folder
if os.path.isdir(_SDK_PATH):
    sys.path.insert(0, _SDK_PATH)
    logger.info(f"Added local hud-sdk path to sys.path: {_SDK_PATH}")
else:
    logger.warning(f"Local hud-sdk path not found at: {_SDK_PATH}, ensure it's at the root of the project.")
# --- End path addition ---

# ---------------------------------------------------------------------------
# HUD SDK setup (local checkout support just like v1 script)
# ---------------------------------------------------------------------------
HUD_AVAILABLE = True
try:
    from hud import gym  # type: ignore
    from hud.task import Task  # type: ignore
    from hud.env.environment import Environment  # type: ignore
    from hud.adapters.claude.adapter import ClaudeAdapter  # type: ignore
except ImportError as e:  # pragma: no cover – clearly surface failure
    HUD_AVAILABLE = False
    logger.error("HUD SDK import failed: %s", e)

# ---------------------------------------------------------------------------
# Payman SDK setup (Python SDK - will not be used for actual payment in this revised flow)
# ---------------------------------------------------------------------------
PAYMAN_AVAILABLE = True
try:
    from paymanai import Paymanai # Will be used for type hinting if needed, not for calls
except ImportError as e:  # pragma: no cover
    PAYMAN_AVAILABLE = False
    logger.info("Payman Python SDK not installed or found. This is okay as payment will be delegated.")

# ---------------------------------------------------------------------------
class ImgurPaymentAgent:
    """Navigate to Imgur in a hud-browser Environment then send a Payman test payment."""

    DEFAULT_IMGUR_URL = "https://imgur.com/a/FKvPe0B"

    def __init__(self) -> None:
        self.imgur_url: str = os.environ.get("IMGUR_ALBUM_URL", self.DEFAULT_IMGUR_URL)
        # PAYMAN_CLIENT_ID and PAYMAN_CLIENT_SECRET will be used by the Node.js script
        # self.payman_client_id: Optional[str] = os.getenv("PAYMAN_CLIENT_ID")
        # self.payman_client_secret: Optional[str] = os.getenv("PAYMAN_CLIENT_SECRET")
        self.payee_id: Optional[str] = os.getenv("PAYMAN_PAYEE_ID") # Fallback

        self.adapter: Optional[ClaudeAdapter] = None
        self.env: Optional[Environment] = None
        self.task: Optional[Task] = None

        # Results bucket
        self.results: Dict[str, Any] = {
            "status": "initialized",
            "imgur_url": self.imgur_url,
            "payee_id": self.payee_id,
        }

    # ---------------------------------------------------------------------
    # Helper – extract album <title>
    # ---------------------------------------------------------------------
    def _fetch_album_title(self) -> str:
        logger.info("Fetching Imgur album HTML to confirm accessibility…")
        try:
            r = requests.get(self.imgur_url, timeout=15)
            r.raise_for_status()
        except Exception as exc:
            logger.error("Failed to fetch Imgur album: %s", exc)
            raise

        title_string: Optional[str] = None
        if BeautifulSoup is not None: # Check if the import succeeded
            soup = BeautifulSoup(r.text, "html.parser")
            title_tag = soup.title
            if title_tag and title_tag.string:
                title_string = title_tag.string.strip()
            title = title_string if title_string else "(no <title>)"
        else:
            # Fallback – regex search for <title>
            m = re.search(r"<title>(.*?)</title>", r.text, re.IGNORECASE | re.DOTALL)
            extracted_title = m.group(1).strip() if m and m.group(1) else None
            title = extracted_title if extracted_title else "(no <title>)"

        logger.info("Imgur album title: %s", title)
        return title

    # ---------------------------------------------------------------------
    # HUD Task definition
    # ---------------------------------------------------------------------
    async def _define_task(self) -> Task:
        logger.info("Defining HUD Task that opens the Imgur album in Chrome…")

        # Simplified setup - just navigate to the URL
        # Based on HUD docs examples: setup=("goto", "url")
        setup_action = ("goto", self.imgur_url)

        # Simple tuple for evaluate - look for "imgur" in the response
        evaluate_config = ("response_includes", ["imgur"]) # type: ignore

        task = Task(
            prompt=f"Navigate to {self.imgur_url} and extract the image title or any payment information shown.",
            gym="hud-browser",
            setup=setup_action,
            evaluate=evaluate_config, # type: ignore
            config={"time_limit_seconds": 180},
        )
        return task

    # ---------------------------------------------------------------------
    async def _init_environment(self):
        # from hud import HUDClient  # This doesn't exist - removed

        api_key = os.getenv("HUD_API_KEY")
        if not api_key: # Basic check, though gym.make might handle it internally
            logger.warning("HUD_API_KEY environment variable not set. SDK might rely on it internally.")

        # Direct gym.make path
        self.task = await self._define_task()
        self.env = await gym.make(self.task)
        self.adapter = ClaudeAdapter()
        logger.info("HUD Environment ready.")

    # ---------------------------------------------------------------------
    async def _run_short_browser_session(self):
        assert self.env, "Environment not initialized"

        # Reset to get initial observation (this also runs the setup actions like "goto")
        try:
            obs, _ = await self.env.reset()
            logger.info("Environment reset successfully (setup actions executed).")
            
            # --- Wait for page to load after goto --- 
            # Increased sleep duration to give Imgur more time to load.
            # A more robust method would be to wait for a specific element.
            logger.info("Waiting for Imgur page to load...")
            await asyncio.sleep(15) # Increased to 15 seconds
            logger.info("Done waiting.")

            # --- Get updated observation after waiting ---
            # Take one step with empty action to get current state after page load
            obs_after_load, _, _, _ = await self.env.step([])
            
            if obs_after_load and obs_after_load.text:
                page_text_fragment = obs_after_load.text[:1000] # Capture more text
                logger.info("Page text after load (sample): %s…", page_text_fragment.replace("\n", " ")[:200])
            else:
                # Fallback to initial observation if the step after load yields no text
                page_text_fragment = obs.text[:1000] if obs and obs.text else ""
                logger.info("Page text from initial reset (sample): %s…", page_text_fragment.replace("\n", " ")[:200])

            self.results["page_observation"] = page_text_fragment

            # --- Attempt to extract a Payman payee ID from the page text ---
            if not self.payee_id:
                match = re.search(r"pd-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", page_text_fragment)
                if match:
                    self.payee_id = match.group(0)
                    logger.info(f"Extracted payee ID from page: {self.payee_id}")

        except Exception as exc:  # pragma: no cover
            logger.error("env.reset() failed: %s", exc)
            return

        # We simply wait a few seconds to give Chrome a chance to load
        # await asyncio.sleep(5) # This was before the step, now handled after reset

        # Take one step with empty action to get current state
        # try:
        #     obs, _, _, _ = await self.env.step([]) # Moved up and enhanced
        #     if obs and obs.text:
        #         page_text_fragment = obs.text[:500]
        #         logger.info("Updated page text: %s…", page_text_fragment.replace("\n", " ")[:120])
        #         self.results["page_observation"] = page_text_fragment
        # except Exception as exc:
        #     logger.error("env.step() failed: %s", exc)

        # Evaluate (built-in)
        try:
            eval_result = await self.env.evaluate()
            self.results["hud_evaluation_result"] = eval_result
            logger.info("HUD evaluation result: %s", eval_result)
        except Exception as exc:
            logger.error("env.evaluate() failed: %s", exc)

    # ---------------------------------------------------------------------
    def _call_nodejs_payman_script(self):
        payment_amount = 0.50
        payment_memo = "HUD-Imgur-Test-V2 (Payment via Node.js)"

        if not self.payee_id:
            logger.error("No payee ID extracted or provided. Cannot call Node.js Payman script.")
            self.results["payman_nodejs_call"] = {"status": "failed", "reason": "missing_payee_id"}
            return

        # Construct path to the Node.js script within moneybench/payman_js_caller
        current_script_dir = Path(__file__).parent    # moneybench/
        payman_js_caller_dir = current_script_dir / "payman_js_caller"
        nodejs_script_path = payman_js_caller_dir / "src" / "sendPaymanPayment.ts"

        if not nodejs_script_path.is_file():
            logger.error(f"Node.js payment script not found at: {nodejs_script_path}")
            self.results["payman_nodejs_call"] = {"status": "failed", "reason": f"Node.js script missing at {nodejs_script_path}"}
            return

        command = [
            "bun", 
            # "run", # Not needed if script path is specified directly for bun
            str(nodejs_script_path),
            self.payee_id,
            str(payment_amount),
            payment_memo
        ]

        logger.info(f"Calling Node.js Payman script: {' '.join(command)}")
        # The CWD for the subprocess should be where the Node.js script can find its .env and node_modules
        logger.info(f"Node.js script working directory: {payman_js_caller_dir}")

        try:
            # Ensure PAYMAN_CLIENT_ID and PAYMAN_CLIENT_SECRET are in moneybench/payman_js_caller/.env
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False, 
                cwd=payman_js_caller_dir # Run the command from within moneybench/payman_js_caller/
            )

            logger.info("Node.js script STDOUT:")
            for line in process.stdout.splitlines():
                logger.info(f"[Node.js STDOUT] {line}")
            
            if process.stderr:
                logger.error("Node.js script STDERR:")
                for line in process.stderr.splitlines():
                    logger.error(f"[Node.js STDERR] {line}")

            if process.returncode == 0:
                logger.info("Node.js Payman script executed successfully (according to exit code).")
                self.results["payman_nodejs_call"] = {"status": "success", "stdout": process.stdout, "stderr": process.stderr}
            else:
                logger.error(f"Node.js Payman script failed with exit code {process.returncode}.")
                self.results["payman_nodejs_call"] = {"status": "failed", "reason": f"Node.js script exit code {process.returncode}", "stdout": process.stdout, "stderr": process.stderr}

        except FileNotFoundError:
            logger.error("'bun' command not found. Please ensure Bun is installed and in your PATH.")
            self.results["payman_nodejs_call"] = {"status": "failed", "reason": "bun_not_found"}
        except Exception as e:
            logger.error(f"Error calling Node.js Payman script: {e}", exc_info=True)
            self.results["payman_nodejs_call"] = {"status": "failed", "reason": str(e)}

    # ---------------------------------------------------------------------
    async def run(self) -> Dict[str, Any]:
        start_ts = time.time()
        self.results["start_time"] = start_ts

        try:
            if not HUD_AVAILABLE:
                raise RuntimeError("HUD SDK is not available in this environment.")

            # 1) Spin up HUD browser task
            await self._init_environment()
            await self._run_short_browser_session()
        except Exception as exc:
            logger.critical("Critical failure during HUD step: %s", exc, exc_info=True)
            self.results["status"] = "hud_error"
            self.results["error"] = str(exc)
        finally:
            if self.env:
                try:
                    await self.env.close()
                except Exception:
                    pass

        # 2) Delegate Payman payment step by calling Node.js script
        self._call_nodejs_payman_script()

        # Wrap up
        self.results["end_time"] = time.time()
        self.results["duration"] = self.results["end_time"] - start_ts

        if self.results.get("status") not in {"hud_error"}:
            self.results["status"] = "completed"

        return self.results


# ---------------------------------------------------------------------------
async def main():
    agent = ImgurPaymentAgent()
    results = await agent.run()

    out_file = Path("hud_imgur_test_v2_results.json")
    try:
        with out_file.open("w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2, default=str)
        logger.info("Results written to %s", out_file)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to write results JSON: %s", exc)
        logger.debug("Raw results object: %s", results)


if __name__ == "__main__":
    asyncio.run(main()) 