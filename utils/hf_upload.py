import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from random import randrange
from transformers import AutoTokenizer, AutoModelForCausalLM


def upload_to_hf(model_dir, model_name):
    """
    Uploads a local model to huggingface.
 
    Args:
        model_dir: local directory for the model
 
    Returns:
        None
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # For loading QLoRA models
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True)

    model.push_to_hub(model_name)
    
    return

upload_to_hf('llama-7-int4-dolly', "qlora_dolly_llama")
    
