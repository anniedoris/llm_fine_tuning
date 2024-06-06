from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.append("utils")
from inference import *

prompt = "What is a good recipe for lemon cake?"
model_response = inference_llm("anniedoris/merged_qlora_dolly_llama", prompt, my_model=True)

print("Response")
print(model_response)

# tokenizer = AutoTokenizer.from_pretrained("anniedoris/merged_qlora_dolly_llama_tokenizer")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# # BitsAndBytesConfig int-4 config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )

# # Load model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(
#     "anniedoris/merged_qlora_dolly_llama",
#     quantization_config=bnb_config,
#     use_cache=False,
#     use_flash_attention_2=False,
#     device_map="auto",
# )