import os
import sys # Import sys
import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv

# --- Add local hud-sdk to path ---
_SDK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hud-sdk'))
if os.path.isdir(_SDK_PATH):
    sys.path.insert(0, _SDK_PATH)
    print(f"Added local hud-sdk path to sys.path: {_SDK_PATH}")
else:
    print(f"Local hud-sdk path not found at: {_SDK_PATH}")
# --- End path addition ---

# Load HUD SDK components
from hud import gym # Use gym module directly
from hud.task import Task
from hud.adapters.claude.adapter import ClaudeAdapter
from hud.env.environment import Environment

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hud_imgur_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hud_imgur_test")

# For double-check fetch of page content
import textwrap
import requests
try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # Lightweight fallback if bs4 isn't installed
    BeautifulSoup = None  # type: ignore

class ImgurTestAgent:
    """Agent for testing basic HUD Task execution with Imgur (using hud.gym.make)."""

    def __init__(self, api_key: Optional[str] = None): # API key might be implicitly used via settings
        # API key is often loaded via dotenv or env vars by the SDK itself
        # self.api_key = api_key
        # self.client = HUDClient(api_key=self.api_key) # Removed client
        self.adapter = ClaudeAdapter()
        self.task: Optional[Task] = None
        # self.run = None # Removed run
        self.env: Optional[Environment] = None

    async def _define_task(self) -> Task:
        """Defines the Imgur navigation Task object dynamically using hud-browser."""
        logger.info("Defining Imgur Test Task for hud-browser (gym.make approach)")
        # Simple QA-style task: ask agent to answer a question
        question_prompt = (
            "Answer this question \"Moneybench is a benchmark which measures the ability of AI agents "
            "to make money in (?)\. The answer is 3 words, all lowercase\"."
        )

        setup_action = None  # No browser navigation required for QA prompt

        target_string = "the real world"

        task = Task(
            # id="imgur-nav-test-gym.make-v2", # Let HUD assign the ID
            prompt=question_prompt,
            gym="hud-browser",
            setup=setup_action, # Pass the tuple directly
            evaluate=("response_includes", [target_string]),  # Evaluate agent's response
            config={
                "time_limit_seconds": 300,
            }
        )
        self.task = task
        return task

    async def initialize_environment(self):
        """Initializes the HUD Environment using hud.gym.make."""
        if not self.task:
             await self._define_task()
        assert self.task is not None, "Task must be defined before initialization."

        logger.info(f"Making HUD Environment directly using hud.gym.make for Task '{self.task.id}'")
        # Use hud.gym.make directly
        self.env = await gym.make(self.task)
        # No separate run object is created in this approach
        # await self.env.wait_for_ready() # This was re-added, remove again
        logger.info("Environment is ready.")

    async def run_agent_loop(self) -> Dict[str, Any]:
        """Contains the main agent interaction loop with timeout (minimal actions)."""
        assert self.env is not None, "Environment not initialized before running agent loop."
        assert self.task is not None and self.task.config is not None, "Task or task config not initialized."

        max_runtime_seconds = self.task.config.get("time_limit_seconds", 300)
        start_time = time.time()
        step_count = 0
        max_steps = 10
        terminated_reason = "max_steps_reached"

        current_text_observation = getattr(self.task, "prompt", "")
        logger.info(f"Starting agent loop for Task ID: {self.task.id}. Time limit: {max_runtime_seconds}s")
        logger.info(f"Initial text observation (prompt): {current_text_observation[:200]}...")

        # Perform an initial environment reset to obtain the first Observation (includes screenshot)
        try:
            initial_obs, _ = await self.env.reset()
            if initial_obs and initial_obs.screenshot:
                logger.info(
                    f"Initial screenshot captured via env.reset() (base64 length: {len(initial_obs.screenshot)})"
                )
            else:
                logger.warning("env.reset() returned no screenshot data.")
        except Exception as e:
            logger.error(f"Error during initial env.reset(): {e}", exc_info=True)

        screenshots: list[str] = []

        # Submit the final answer immediately via ResponseAction
        from hud.adapters.common.types import ResponseAction
        try:
            # Send the response action to the environment
            response_action = ResponseAction(text="the real world")
            await self.env.step(response_action)  # type: ignore[arg-type]
            terminated_reason = "response_submitted"
            logger.info("Submitted ResponseAction with answer 'the real world'.")
        except Exception as e:
            logger.error(f"Error submitting ResponseAction: {e}", exc_info=True)

        # Optionally collect one screenshot after submission for logging
        try:
            observation, _, _, _ = await self.env.step([])
            if observation and observation.screenshot:
                screenshots.append(observation.screenshot)
        except Exception:
            pass

        # No iterative loop needed; we skip while loop
        step_count = 1

        final_time = time.time()
        duration = final_time - start_time
        logger.info(f"Agent loop finished. Reason: {terminated_reason}. Duration: {duration:.2f}s")

        return {
            "start_time": start_time,
            "end_time": final_time,
            "duration": duration,
            "steps_taken": step_count,
            "termination_reason": terminated_reason,
            "screenshots": screenshots,
        }

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Runs the full flow: define task, make env, agent loop, evaluate, close."""
        full_results: Dict[str, Any] = {"status": "started", "error": None}
        try:
            # 1. Define Task & Create Environment directly
            await self._define_task() # Defines self.task
            await self.initialize_environment() # Creates self.env using gym.make
            if self.task:
                 full_results["task_id"] = self.task.id
            else:
                 full_results["task_id"] = "Task not defined"
            full_results["env_initialized"] = True

            # 2. Run Agent Interaction Loop
            loop_results = await self.run_agent_loop()
            full_results.update(loop_results)
            full_results["loop_completed"] = True

            # 3. Run built-in evaluation configured in Task
            logger.info("Calling env.evaluate() for built-in page_contains evaluation...")
            assert self.env is not None, "Environment unexpectedly None during evaluation"
            evaluation_result = await self.env.evaluate()
            full_results["evaluation_result"] = evaluation_result
            full_results["final_score"] = evaluation_result.get("score") if isinstance(evaluation_result, dict) else None
            logger.info(f"Evaluation result: {evaluation_result}")

            # Extra confirmation in logs
            if isinstance(evaluation_result, dict):
                reward = evaluation_result.get("reward") or evaluation_result.get("score")
                success = bool(reward and reward >= 1.0)

                if success:
                    logger.info("CONFIRMED: Target string successfully detected on the page [SUCCESS]")
                    print("[HUD TEST] Target string detected — evaluation passed [SUCCESS]")
                else:
                    logger.warning("Target string NOT detected — evaluation failed [FAIL]")
                    print("[HUD TEST] Target string NOT detected — evaluation failed [FAIL]")

                # --- Double-check: fetch page HTML locally and print paragraph ---
                try:
                    target = "In contrast, Moneybench runs in the real world. This allows us to study agents in-situ"
                    response = requests.get("https://moneybench.github.io/moneybench/#home", timeout=15)
                    response.raise_for_status()
                    paragraph_text = None
                    if BeautifulSoup:
                        soup = BeautifulSoup(response.text, "html.parser")
                        for p in soup.find_all("p"):
                            if target in p.get_text():
                                paragraph_text = p.get_text(strip=True)
                                break
                    else:
                        # Fallback: simple string search in raw HTML
                        idx = response.text.find(target)
                        if idx != -1:
                            # Extract ~200 chars around
                            start = max(idx - 100, 0)
                            end = idx + len(target) + 100
                            paragraph_text = response.text[start:end]

                    if paragraph_text:
                        logger.info("Paragraph containing target string (local fetch):\n%s", textwrap.fill(paragraph_text, 120))
                        print("\n[DOUBLE-CHECK] Paragraph containing target string:\n" + textwrap.fill(paragraph_text, 120))
                    else:
                        logger.warning("Could not locate paragraph containing the target string via local fetch.")
                except Exception as fetch_err:
                    logger.error("Error during local page fetch for double-check: %s", fetch_err, exc_info=True)

            full_results["status"] = "completed"

        except Exception as e:
            logger.critical(f"Critical error during evaluation flow: {e}", exc_info=True)
            full_results["status"] = "error"
            full_results["error"] = str(e)
        finally:
            # 4. Close Environment
            if self.env:
                logger.info("Closing HUD environment.")
                try:
                    await self.env.close()
                except Exception as close_err:
                    logger.error(f"Error closing environment: {close_err}")
                    if full_results["status"] != "error":
                        full_results["status"] = "error_closing"
                        # Ensure error key exists before appending
                        existing_error = full_results.get("error", "")
                        full_results["error"] = f"{existing_error} | Close Error: {close_err}"

        return full_results

async def main():
    # Explicitly load .env file first
    dotenv_path = find_dotenv()
    if dotenv_path:
        logger.info(f"Loading .env file from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True) # Override ensures .env takes precedence
    else:
        logger.warning(".env file not found. Relying on system environment variables.")

    # Load and mask API key for verification
    hud_api_key = os.getenv("HUD_API_KEY")

    if not hud_api_key or "YOUR_" in hud_api_key:
        logger.critical("HUD_API_KEY environment variable not set, is default, or could not be loaded.")
        return
    else:
        # Print masked key
        masked_key = f"{hud_api_key[:5]}...{hud_api_key[-5:]}" if len(hud_api_key) > 10 else hud_api_key
        logger.info(f"Loaded HUD_API_KEY (masked): {masked_key}")

    logger.info("--- Starting HUD Imgur Test Agent (gym.make approach) ---")
    # Pass the explicitly loaded key, although gym.make might not use it directly
    agent = ImgurTestAgent(api_key=hud_api_key)

    results = await agent.run_full_evaluation()

    results_file = "hud_imgur_test_results.json"
    logger.info(f"--- Evaluation Ended: Status: {results.get('status')} ---")
    logger.info(f"Attempting to save results to {results_file}")
    try:
        def default_serializer(obj):
            if callable(obj):
                 return f"<function {obj.__name__}>"
            # Add check for Task object if needed, though Task might be serializable
            if isinstance(obj, Task):
                 return f"<Task id={obj.id}>" # Basic representation
            try:
                 json.dumps(obj)
                 return obj
            except TypeError:
                 return str(obj)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=default_serializer)
        logger.info(f"Results successfully saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}", exc_info=True)
        logger.error(f"Raw results data:\n{results}")

    logger.info("--- HUD Imgur Test Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True) 