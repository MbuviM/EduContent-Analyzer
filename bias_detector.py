# Creating a gender bias detetor in text data using Llama 3.2 model
## Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Load datasets
anti = load_dataset("uclanlp/wino_bias", "type2_anti")
pro = load_dataset("uclanlp/wino_bias", "type2_pro")

# Load Llama 3.2 model and tokenizer from Hugging Face Hub
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

