########################################################
# Kwansi is a wrapper for DSPy that makes its optimizer easier to use.
# This is an example implementation of how to use Kwansi to optimize a task.
#
# To get started:
# 1. Read the README.md for the explanation of the example below.
# 2. Find more information about kwansi at https://github.com/lordamp/kwansi
# 3. Find more information about DSPy at https://dspy-docs.vercel.app/docs 
#
# Make sure to install the kwansi package:
# pip install git+https://github.com/lordamp/kwansi.git
# or via requirements.txt: pip install -r requirements.txt
# You will also need to create an .env file with your OpenAI API key:
# OPENAI_API_KEY=<your-key>
########################################################


import dspy
import os
from dotenv import load_dotenv
import json

# Import the functions from the kwansi package
from kwansi.data_preparation import prepare_examples
from kwansi.evaluation import create_evaluator
from kwansi.optimizer_handling import run_optimizer, save_optimized_model
from kwansi.task_creation import create_task
from kwansi.testing import test_model

# load the environment variables (especially the LLM API key)
load_dotenv()

# load example files (assessors, task, metrics, custom combiner)
from components.assessors import Assess_Interestingness, Assess_StyleAppropriateness
from components.task import TweetCreatorSignature
from components.metrics import length_metric, hashtag_count_metric
from components.custom_combiner import custom_combine

# Load the data - make sure to transform to JSON format
with open('data/example_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define the input fields in the JSON file
input_fields = {
    'data_key': 'tweet_instructions',  # the key in the JSON file that contains the data
    'fields': ['topic', 'details']  # the fields in the JSON file that are the input to the task
}

# Prepare examples (i.e. the training data)
tweet_examples = prepare_examples(data, input_fields, n_samples=50)

# Create the task (i.e. the main prompt we're trying to optimize - defined in signatures/task.py)
TweetCreator = create_task(TweetCreatorSignature, 'ChainOfThought')
tweet_creator = TweetCreator()

# Define the metric (i.e. the various evaluators of the task)
tweet_evaluator = create_evaluator(
    # Assessors are the prompt-based evaluators of the task (defined in signatures/assessors.py)
    assessors=[
        ('Interestingness', Assess_Interestingness, {'tweet': 'tweet'}, (0, 10)),
        ('Style_Appropriateness', Assess_StyleAppropriateness, {'tweet': 'tweet', 'topic': 'topic'}, (0, 10))
    ],
    # Additional metrics are traditional binary metrics (defined in metrics/metrics.py)
    additional_metrics=[
        ('Length_Check', length_metric),
        ('Hashtag_Count', hashtag_count_metric)
    ],
    # The combine method is the way we combine the assessor scores and additional metrics into a single score
    # "additive" means we add the scores, "multiplicative" means we multiply the scores
    # you can also import a custom combiner like the one in metrics/custom_combiner.py
    combine_method="multiplicative",
    # The threshold is the minimum score for the metric to be considered (barely) passing
    threshold=0.25
)

# define the language model DSPy will use
dspy.settings.configure(lm=dspy.LM(
    model='gpt-4o-mini',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=1024
))

# Run the optimizer (i.e. the process of optimizing the task)
# in this example, we use the BootstrapFewShot optimizer,
# but you can choose from a variety of optimizers:
# BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPROv2
# and MIPROv2ZeroShot

optimized_tweet_creator, optimizer_type = run_optimizer(
    optimizer_type='BootstrapFewShotWithRandomSearch',
    evaluator=tweet_evaluator,
    student=tweet_creator,
    trainset=tweet_examples,
)

# Save the optimized model - now including the optimizer type in the filename
save_optimized_model(optimized_tweet_creator, optimizer_type, folder='output', name='tweet_creator')

# Test the Optimized Program (use verbose=True for more details)
print("Short Test Output:")
test_model(
    model=optimized_tweet_creator,
    test_data=tweet_examples,
    n_tests=3,
    input_fields=['topic', 'details'],
    output_field='tweet',
    evaluator=tweet_evaluator,
    verbose=True
)