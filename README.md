# Moneybench

Moneybench is a benchmark for measuring AI agent capabilities in real-world financial interactions. The project provides a framework for testing AI agents' ability to perform financial transactions safely and effectively.

## Overview

Moneybench provides:
- A standardized environment for testing AI agents with financial APIs
- Integration with Stripe for payment processing and financial operations
- Multi-agent evaluation capabilities with different strategies
- Logging and evaluation metrics
- Safety guidelines and constraints

## Development Status

This project now uses standard package installations rather than requiring local repositories. It integrates with the Stripe Agent Toolkit for enhanced financial operations.

## Project Structure
```
.
├── moneybench/                  # Core package
│   ├── test_agent.py            # Single agent implementation
│   ├── multi_agent_eval.py      # Multi-agent evaluation
│   ├── moneybench_task.py       # Task definition
│   └── logs/                    # Evaluation logs
├── agent-toolkit/               # Optional - automatically added to path if present
├── .env                         # Environment variables (not in repo)
├── .env.example                 # Example environment file
└── .gitignore                   # Git ignore patterns
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/menhguin/moneybench.git
cd moneybench
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install stripe-agent-toolkit  # For enhanced Stripe functionality
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Run tests:
```bash
# Single agent test
python moneybench/test_agent.py

# Multi-agent evaluation
python moneybench/multi_agent_eval.py
```

## Configuration

The following environment variables are required:
- `STRIPE_SECRET_KEY`: Your Stripe API key. While in development, we're using Secret Key and Test Mode.
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI models)
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Claude models)
- `INSPECT_EVAL_MODEL`: Model to use for evaluation (e.g., openai/gpt-4-mini)

See `.env.example` for all configuration options.

## Wallet Methods

MoneyBench supports two approaches for multi-agent evaluations:

### 1. Shared Wallet with Agent-Specific Metadata
- All agents use the same Stripe API key (same wallet)
- Actions are distinguished using agent-specific metadata
- Simpler setup, easier to track total activity
- Example in `multi_agent_eval.py`:
  ```python
  intent = stripe.PaymentIntent.create(
      # ...
      metadata={
          "agent_id": self.agent_id  # Distinguishes which agent created the payment
      },
      api_key=self.api_key
  )
  ```

### 2. Different Wallets for Different Agents
- Each agent uses a different Stripe API key (different wallets)
- Complete isolation between agents
- Requires managing multiple Stripe accounts
- To implement, initialize each agent with its own API key:
  ```python
  agent1 = StripeAgent(agent_id="1", api_key=api_key_for_agent1, ...)
  agent2 = StripeAgent(agent_id="2", api_key=api_key_for_agent2, ...)
  ```

## Stripe Agent Toolkit Integration

MoneyBench now integrates with the Stripe Agent Toolkit, providing enhanced capabilities:
- Creating products, prices, and payment links
- Managing customers and invoices
- Handling refunds and balance operations

The toolkit is automatically detected and used if installed:
```python
try:
    from stripe_agent_toolkit.openai.toolkit import StripeAgentToolkit
    # Toolkit is available
except ImportError:
    # Fallback to basic functionality
```

## Multi-Agent Evaluation

The `multi_agent_eval.py` script provides a framework for comparing different agent strategies:
- Sequential execution within a shared time window
- Metrics for comparing performance (accumulation rate, total increase)
- Transcript logging for detailed analysis
- Example strategies included:
  - Aggressive revenue generation
  - Steady, sustainable growth

## Safety and Guidelines

Moneybench is designed for research and testing purposes only. When using this framework:
- Only use test API keys
- Follow all API provider terms of service
- Do not engage in harmful or illegal activities
- Monitor agent behavior closely

## License

MIT License - See LICENSE file for details 
