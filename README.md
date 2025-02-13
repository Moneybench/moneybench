# MoneyBench

MoneyBench is a benchmark for measuring AI agent capabilities in real-world financial interactions. The project provides a framework for testing AI agents' ability to perform financial transactions safely and effectively.

## Overview

MoneyBench provides:
- A standardized environment for testing AI agents with financial APIs
- Integration with Stripe for payment processing
- Logging and evaluation metrics
- Safety guidelines and constraints

## Development Status

⚠️ **Note**: This project is currently under development. For fast iteration, it assumes both repositories are present in the same directory as `moneybench`:
- `inspect_ai`: Core agent framework
- `inspect_evals`: Evaluation framework

 See the intended structure below.

## Project Structure
```
.
├── moneybench/          # Core package
│   ├── test_agent.py    # Test implementation
│   └── moneybench_task.py
├── inspect_ai/          # Required local dependency
├── inspect_evals/       # Required local dependency
├── .env                 # Environment variables (not in repo)
├── .env.example         # Example environment file
└── .gitignore          # Git ignore patterns
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
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Run tests:
```bash
python moneybench/test_agent.py
```

## Configuration

The following environment variables are required:
- `STRIPE_SECRET_KEY`: Your Stripe API key
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI models)
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Claude models)
- `INSPECT_EVAL_MODEL`: Model to use for evaluation (e.g., openai/gpt-4)

See `.env.example` for all configuration options.

## Safety and Guidelines

MoneyBench is designed for research and testing purposes only. When using this framework:
- Only use test API keys
- Follow all API provider terms of service
- Do not engage in harmful or illegal activities
- Monitor agent behavior closely

## License

MIT License - See LICENSE file for details 
