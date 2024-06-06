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
        low_cpu_mem_usage=True)

    # Need to merge LoRA and base model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("merged_model",safe_serialization=True)
    tokenizer.save_pretrained("merged_model")

    # push merged model to the hub
    merged_model.push_to_hub("anniedoris/" + model_name)
    tokenizer.push_to_hub("anniedoris/" + model_name + "_tokenizer")

    # model.config.to_json_file("config.json")
    # model.push_to_hub(model_name)
    
    return

upload_to_hf('llama-7-int4-dolly', "merged_qlora_dolly_llama")
    
