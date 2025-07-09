# Moneybench – Real-World Agent Benchmark

> **Status:** Active development (May 2025) VERY WIP
>
> Moneybench is an experimental benchmark that measures how well autonomous AI agents can *make money* in the real world under time pressure and with limited information.  
> The current milestone focuses on a single end-to-end demo that combines:
>
> 1. **HUD SDK** – a browser/control environment for agent: https://github.com/hud-evals/hud-sdk
> 2. **Payman** – an API that allows programmatic peer-to-peer cash transfers.

---

## 1  High-Level Demo Flow

The showcase script `moneybench/hud_imgur_test_v2.py` performs the following steps:

1. Start a **hud-browser** environment (Chrome in a cloud VM).  
2. Visit a public Imgur album that *contains (or pretends to contain)* a Payman *payee ID* such as `pd-…`.  
3. Wait until the page is fully loaded, grab the raw page text and (optionally) parse out that `payee_id` with a regex.
4. Spawn a **Node (Bun)** subprocess that executes `payman_js_caller/src/sendPaymanPayment.ts` and sends **USD 0.50** to that payee using the Payman Client-Credentials OAuth flow.  
5. Save a JSON result bundle that includes HUD evaluation metrics, stdout/stderr from the Node script, timings, and any errors.

If everything is configured correctly you will see the following in `hud_imgur_test_v2_results.json`:
```jsonc
{
  "status": "completed",
  "hud_evaluation_result": { "reward": 1.0, "details": … },
  "payman_nodejs_call": { "status": "success", "stdout": "{\"id\":…}", … }
}
```

---

## 2  Prerequisites

| Category | Requirement | Why it is needed |
|----------|-------------|------------------|
| **Accounts** | • HUD account + API key  
• Payman developer account (Client ID & Secret) | Authenticating the HUD browser environment and making Payman payments |
| **OS / Shell** | Any OS with Python ≥ 3.11 **and** [Bun](https://bun.sh) ≥ 1.1 installed & on the `PATH`.<br>Examples assume **Windows 10+ PowerShell**. | Python drives HUD; Bun runs the Node payment script |
| **Python tooling** | • [uv](https://github.com/astral-sh/uv) package manager *(faster `pip`)*  
• A virtual-env (`python -m venv .venv`) | Isolates Python deps |
| **Node tooling** | Bun installs and runs the TypeScript Payman SDK automatically (`bun install`, `bun <script>`). | Fast startup & native TypeScript support |

### API Keys & Secrets

All credentials are obtained from your HUD & Payman dashboards and stored **locally** in `.env` files (never commit them!):

```
# at workspace root (for Python)
HUD_API_KEY="hud_sk_live_…"
IMGUR_ALBUM_URL="https://imgur.com/a/FKvPe0B"      # optional override
PAYMAN_PAYEE_ID="pd-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"  # optional fallback

# inside moneybench/payman_js_caller/.env (for Node)
PAYMAN_CLIENT_ID="pm_live_client_…"
PAYMAN_CLIENT_SECRET="pm_live_secret_…"
```

---

## 3  Step-by-Step Setup

```powershell
# 1. Clone & enter the repo
PS> git clone https://github.com/your-fork/inspect-moneybench-10022025.git
PS> cd inspect-moneybench-10022025

# 2. Python venv + dependencies (≈ 30 s with uv)
PS> python -m venv .venv ; .\.venv\Scripts\Activate.ps1
PS> uv pip install -r requirements.txt

# 3. Install Bun (skip if already on PATH)
PS> iwr https://bun.sh/install -UseBasicParsing | iex           # PowerShell installer
PS> bun --version                                              # should print a version

# 4. Node deps for the payment caller
PS> cd moneybench\payman_js_caller
PS> bun install                                                # installs @paymanai/payman-ts etc.
PS> cd ../..

# 5. Create both .env files (see previous section).
```

---

## 4  Running the Demo

```powershell
PS> .\.venv\Scripts\Activate.ps1      # if not already active
PS> python -B moneybench/hud_imgur_test_v2.py
```

Output locations:

* **logs/**`hud_imgur_test_v2.log` – verbose log (browser steps, stdout/stderr from Node, tracebacks, …)
* **hud_imgur_test_v2_results.json** – structured summary for post-processing / scoring

If the run is fully successful you should see something like:
```text
2025-05-02 21:07:42 – INFO – Node.js Payman script executed successfully (according to exit code).
```

---

## 5  Walk-through of `hud_imgur_test_v2.py`

| Phase | What happens (simplified) | Key source lines |
|-------|---------------------------|------------------|
| **Init** | Imports, logging, pulls `.env`, adds `hud-sdk` to `sys.path`. | 30-80 |
| **HUD Task** | Build a `Task` that tells HUD to `goto <imgur_url>` and later checks that the response *includes* the word `imgur`. | 120-150 |
| **Environment** | `env = await gym.make(task)`. This spins up a remote Chrome via HUD Cloud, executes the `goto`, and returns an observation (DOM text, screenshot). | 160-200 |
| **Wait & Scrape** | `await asyncio.sleep(15)` gives the page time to load. The script reads `obs_after_load.text`, keeps the first 1 000 chars, and attempts a regex: `r"pd-[0-9a-f-]{36}"`. | 200-240 |
| **Evaluation** | `await env.evaluate()` re-runs the task’s `evaluate` tuple – essentially an assertion that “imgur” is present. | 245-260 |
| **Close HUD** | `await env.close()` tears down the VM to avoid billing. | 270 |
| **Payment** | `_call_nodejs_payman_script()` builds a Bun command:  
`bun sendPaymanPayment.ts <payeeId> 0.5 "memo"`  
and runs it with `cwd` = `payman_js_caller`. | 280-340 |
| **Result dump** | A dictionary with timings and payman response is JSON-serialized to `hud_imgur_test_v2_results.json`. | 350-370 |

### 5.1  Why delegate to Node?

The Python Payman SDK (current `v2.7.x`) only supports API-Secret auth which is **insufficient** for the *payment* endpoint that requires an *access token*. Payman’s TypeScript SDK handles the Client-Credentials OAuth dance automatically, so we simply call it from Python instead of re-implementing the flow.

---

## 6  HUD 101 (for absolute beginners)

* **HUD Cloud** gives each task a *gym* – an executable sandbox.  For browser work we use the `hud-browser` gym which boots a headless Chrome inside a VM that the agent controls via the HUD API.
* A **Task** bundles:  
  • `prompt` – natural-language instructions to the agent model.  
  • `setup` – deterministic actions (e.g. `("goto", "https://…")`).  
  • `evaluate` – automated tests (e.g. `( "response_includes", ["foo"] )`).
* The Python helper `hud.gym.make(task)` returns an **Environment** that conforms to the OpenAI Gym API (`reset`, `step`, etc.).

---

## 7  Payman 101

* **Payman** is a programmable wallet for sending small P2P payments (< $5) with near-zero fees.  
* Auth follows OAuth 2 *Client Credentials* (→ **access token**).  
* SDKs:  
  • **`@paymanai/payman-ts`** – full support (used here).  
  • **`paymanai` Python** – limited (works for read-only endpoints, not payments).
* **Payee ID** (`pd-…`) is analogous to an email for payments. Anyone can send funds to that identifier; only the owner can withdraw.

---

## 8  Project Directory Snapshot (relevant bits)

```text
inspect-moneybench-10022025/
├─ moneybench/
│  ├─ hud_imgur_test_v2.py          # Python orchestrator
│  ├─ payman_js_caller/
│  │  ├─ src/sendPaymanPayment.ts   # Bun/TypeScript payment helper
│  │  ├─ package.json
│  │  ├─ tsconfig.json
│  │  └─ .env                       # Payman credentials (NOT COMMITTED)
│  └─ README.md                     # ← you are here
├─ hud-sdk/                         # optional, local checkout of HUD
├─ requirements.txt
└─ .env                             # HUD_API_KEY etc.
```

---

## 9  Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: hud` | `hud-sdk` not on `PYTHONPATH` | The script automatically prepends `../hud-sdk`; ensure the folder exists or `pip install hud-python`.
| `"bun" is not recognized` | Bun not installed / not in `PATH` | Re-install Bun and **open a new terminal** (Windows: logout/login or run `refreshenv`).
| Payman "401 Unauthorized – Missing x-payman-access-token" | Node `.env` missing or wrong client credentials | Double-check `PAYMAN_CLIENT_ID` & `PAYMAN_CLIENT_SECRET`.
| HUD eval score 0 | Page didn’t load in time | Increase `await asyncio.sleep(…)` or verify network connectivity.

---

## 10  Contributing

1. Fork & branch off `main`.
2. Create reproducible test cases – use the JSON result bundle.
3. Ensure `ruff`, `mypy`, and `pytest` pass (`uv pip install -r requirements-dev.txt`).
4. Open a PR with a concise description and link to your result bundle.

---

## 11  License

MIT – see `LICENSE`.
