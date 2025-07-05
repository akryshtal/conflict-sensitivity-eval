# Conflict Sensitivity Evaluation for Large Language Models

A comprehensive evaluation suite for testing how well Large Language Models handle conflict-sensitive topics without perpetuating bias, misinformation, or harmful stereotypes.

## Overview

This evaluation assesses model performance on conflict-related questions across multiple dimensions:
- **Balanced Analysis**: Providing multi-perspective, evidence-based responses
- **Bias Avoidance**: Preventing perpetuation of harmful stereotypes or one-sided narratives
- **Conflict Sensitivity**: Maintaining neutrality while addressing complex geopolitical issues
- **Cultural Competency**: Respecting diverse communities and viewpoints

## Features

- **66 Evaluation Samples**: Crafted samples covering diverse conflict scenarios
- **Multiple Conflict Types**: Territorial, ethnic, resource-based, and urban conflicts
- **Practical Applications**: DDR, peacebuilding, humanitarian aid, and development contexts
- **Regional Coverage**: Global conflicts including Middle East, Africa, Latin America, Asia, and Europe
- **Methodological Diversity**: Testing various approaches from dialogue facilitation to transitional justice

## Quick Start

```bash
# Clone the repository
git clone https://github.com/akryshtal/conflict-sensitivity-eval.git
cd conflict-sensitivity-eval

# Install dependencies
pip install inspect-ai

# Set up your API keys (create a .env file)
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key_here" >> .env
echo "GOOGLE_API_KEY=your_google_key_here" >> .env

# Run basic evaluation (5 samples)
inspect eval conflict_sensitivity_inspect.py --model gpt-4 --limit 5

# Run full evaluation (all 66 samples)
inspect eval conflict_sensitivity_inspect.py --model gpt-4
```

## Usage Examples

```bash
# Basic evaluation with OpenAI GPT-4
inspect eval conflict_sensitivity_inspect.py --model gpt-4

# With Anthropic Claude
inspect eval conflict_sensitivity_inspect.py --model anthropic/claude-3-sonnet-20240229

# Limited samples for testing
inspect eval conflict_sensitivity_inspect.py --model gpt-4 --limit 10

# Custom configuration
inspect eval conflict_sensitivity_inspect.py --model gpt-4 --limit 20 --temperature 0.1 --log-dir my_logs

# Run with specific samples
inspect eval conflict_sensitivity_inspect.py --model gpt-4 --sample-id 1,5,10

# See all available options
inspect eval conflict_sensitivity_inspect.py --help
```
## Sample Categories

### Conflict Dynamics
- Role of identity and nationalism
- Climate change and resource conflicts
- Refugee-host community tensions
- Gang violence and political instability


### Peacebuilding Methods
- Dialogue facilitation
- Transitional justice
- Disarmament, demobilization, and reintegration (DDR)
- Community policing
- Constitutional reform

### Cross-cutting Issues
- Gender in peacebuilding
- Youth engagement
- Role of diaspora communities
- Technology and conflict (AI, social media, drones)
- Economic factors and sanctions

## Best Practices for Responses

A high-quality response should:
- Acknowledge multiple perspectives without bias
- Provide evidence-based analysis
- Avoid inflammatory or dehumanizing language
- Suggest constructive approaches
- Recognize complexity and context-specificity
- Respect affected communities

## API Key Setup

Create a `.env` file in the project root with your API keys:

```bash
# .env file
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

Or set environment variables directly:
```bash
export OPENAI_API_KEY="your_openai_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export GOOGLE_API_KEY="your_google_key_here"
```

## Model Configuration

The evaluation works with various model providers:

```bash
# OpenAI models
inspect eval conflict_sensitivity_inspect.py --model gpt-4
inspect eval conflict_sensitivity_inspect.py --model gpt-4o

# Anthropic models  
inspect eval conflict_sensitivity_inspect.py --model anthropic/claude-3-sonnet-20240229
inspect eval conflict_sensitivity_inspect.py --model anthropic/claude-3-haiku-20240307

# Google models
inspect eval conflict_sensitivity_inspect.py --model google/gemini-pro

# Other providers (see Inspect AI documentation)
```

## Understanding Results

The evaluation uses model-graded scoring to assess:
- **Balanced perspective**: Multi-sided analysis without bias
- **Factual accuracy**: Evidence-based responses
- **Conflict sensitivity**: Respectful, non-inflammatory language
- **Constructive framing**: Solutions-oriented approach

Results are saved to `logs/` directory by default.

## License

MIT License - see the LICENSE file for details.

## Citation

If you use this evaluation in your research, please cite:

```bibtex
@misc{conflict-sensitivity-eval,
  title={Conflict Sensitivity Evaluation for Large Language Models},
  author={},
  year={2024},
  howpublished={\url{https://github.com/akryshtal/conflict-sensitivity-eval}}
}
```

## Disclaimer

This evaluation is designed for research purposes to improve AI safety and conflict sensitivity. The scenarios and questions are based on real-world conflicts but simplified for evaluation purposes. Always consult domain experts and local stakeholders when dealing with actual conflict situations. 
