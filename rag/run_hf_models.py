from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
sys.path.append("utils")
from inference import *


result = inference_llm("lmsys/vicuna-7b-v1.5", "What is your name?")
print(result)

# model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map="auto", load_in_4bit=True)

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
# model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")