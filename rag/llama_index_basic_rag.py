import os
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.core import set_global_tokenizer
from llama_index.core import Settings

from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)

# Huggingface token
# HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

Settings.tokenizer = AutoTokenizer.from_pretrained("anniedoris/merged_qlora_dolly_llama_tokenizer").encode
model = HuggingFaceLLM(model_name = "anniedoris/merged_qlora_dolly_llama")

response = model.complete("What is a good recipe for lemon cake?")
print("Model response")
print(response)