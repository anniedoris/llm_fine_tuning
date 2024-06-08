from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
sys.path.append("utils")
from inference import *
import transformers
import torch


# result = inference_llm("NousResearch/Llama-2-7b-chat-hf", "What is your name?", cuda=False)
# result = inference_llm("lmsys/vicuna-7b-v1.5", "What is your name?", cuda=False)
# print(result)

# model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map="auto", load_in_4bit=True)

# tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", padding_side="left")
# model_inputs = tokenizer(["What is your name"], return_tensors="pt")

# Inference pipeline for llama 2 chat
model = "NousResearch/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

model = AutoModelForCausalLM.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.float16,
    max_new_tokens = 512,
    device_map="auto",
)

sequences = pipeline(
    "I'm looking to make a lemon cake. Could you please give me a recipe for a good lemon cake?",
    do_sample=True,
)

print(sequences[0].get("generated_text"))

# Inference pipeline for llama 2 no chat
model = "NousResearch/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

model = AutoModelForCausalLM.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.float16,
    max_new_tokens = 512,
    device_map="auto",
)

sequences = pipeline(
    "I'm looking to make a lemon cake. Could you please give me a recipe for a good lemon cake?",
    do_sample=True,
)

print(sequences[0].get("generated_text"))