from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
sys.path.append("utils")
from inference import *


# result = inference_llm("NousResearch/Llama-2-7b-chat-hf", "What is your name?", cuda=False)
# result = inference_llm("lmsys/vicuna-7b-v1.5", "What is your name?", cuda=False)
# print(result)

# model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map="auto", load_in_4bit=True)

# tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", padding_side="left")
# model_inputs = tokenizer(["What is your name"], return_tensors="pt")

