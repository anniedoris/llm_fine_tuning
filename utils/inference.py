import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from random import randrange
from transformers import AutoTokenizer, AutoModelForCausalLM


def inference_llm(model_dir, prompt, peft_model=False):
    """
    Runs inference on an LLM.
 
    Args:
        model_dir: either local directory or huggingface directory for the model
        prompt: the prompt to ask the llm
        peft_model: whether the model is peft or not, different loading methods
 
    Returns:
        response: model's response to the prompt
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # For loading QLoRA models
    if peft_model:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)
        outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    
    # For loading pre-trained/non-lora models
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda")
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs)
        outputs = tokenizer.decode(outputs[0])
    return outputs
    
