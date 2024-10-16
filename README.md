# Kwansi Example Implementation

This repository contains an example implementation of Kwansi, a wrapper for [DSPy](https://dspy-docs.vercel.app/docs) that makes its optimizers easier to use. You can find Kwansi here: https://github.com/baloise/kwansi, which also includes a documentation about its core functions.

## Overview

This example demonstrates how to use Kwansi to optimize a tweet creation task. It covers:
- Setting up the environment
- Preparing training data
- Creating a task
- Defining metrics
- Running the optimizer
- Testing the optimized model

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/lordamp/kwansi-example.git
   cd kwansi-example
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

1. Run the main script:
   ```
   python example_implementation.py
   ```

2. For a step-by-step walkthrough, see the [`example_implementation.ipynb`](https://github.com/baloise/kwansi_example/blob/main/example_implementation.ipynb) Jupyter notebook.

## File Structure

- `example_implementation.py`: The example implementation of Kwansi. Run this to execute the whole example.
- `example_implementation.ipynb`: The step-by-step walkthrough of the example implementation as a Jupyter notebook.
- `components/`: Custom task and evaluator definitions
    - `task.py`: The task instructions we're trying to optimize
    - `assessors.py`: The assessor evaluators used to evaluate the task (LLM-based evaluators)
    - `metrics.py`: The metrics used to evaluate the task (binary metrics)
    - `custom_combiner.py`: A custom combiner function for the evaluators
- `data/`: Example data for the task
    - `example_data.json`: The prepared data used for training the optimizer
- `outputs/`: Directory for storing the compiled instructions (empty at the start)


## Further Documentation

For more detailed information about Kwansi, please visit the [Kwansi documentation](https://github.com/lordamp/kwansi). If you want to learn more about the underlying DSPy framework, please visit the [DSPy documentation](https://dspy-docs.vercel.app/docs).
