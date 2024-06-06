import sys
sys.path.append("utils")
from inference import *
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
sample = dataset[randrange(len(dataset))]

# Format the prompt
prompt = f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.
 
### Input:
{sample['response']}
 
### Response:
"""
# prompt = "What is a good recipe for lemon cake?"

# Run inference for ift model
model_response = inference_llm("llama-7-int4-dolly", prompt, peft_model=True)

# Run inference for pretrained model
pretrained_model_response = inference_llm("NousResearch/Llama-2-7b-hf", prompt)

print("\n")
print("INSTRUCTION TUNED MODEL")
print(f"Prompt:\n{sample['response']}\n")
print(f"Ground truth:\n{sample['instruction']}\n")
print(f"Generated instruction ift:")
print(model_response)

print("\n")
print("PRETRAINED MODEL")
print(f"Prompt:\n{sample['response']}\n")
print(f"Ground truth:\n{sample['instruction']}\n")
print(f"Generated instruction pretrained:")
print(pretrained_model_response)