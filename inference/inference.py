from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
sys.path.append("utils")
from inference import *
import transformers
import torch
from torch import cuda
from transformers import GenerationConfig
from peft import AutoPeftModelForCausalLM

# Referenced this video and associated colab: https://www.youtube.com/watch?v=Z6sCl6abJj4

def inference_hf_model(model_name, input_prompt, max_new_toks=250):
    """
        Gets a inference response from a huggingface model based on a prompt.

        Parameters:
            model_name: name of the huggingface model
            input_prompt: prompt to ask the huggingface model
            max_new_tokens: number of tokens you'd like the model to generate

        Returns:
            Model's response
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer = tokenizer,
    torch_dtype=torch.float16,
    max_new_tokens = max_new_toks,
    device_map="auto"
    )

    # There are other traditional LLM inference settings that can be modified here
    sequences = model_pipeline(
            input_prompt,
            do_sample=True
    )

    # TODO: play with other parameters settings for inference? Here's an example of some samples below
#     # sequences = model_pipeline(
#     #     prompt,
#     #     do_sample=True,
#     #     top_k=10,
#     #     num_return_sequences=1,
#     #     eos_token_id=tokenizer.eos_token_id,
#     #     max_length=256,
#     # )

    model_response = sequences[0]['generated_text']
    model_response_before_strip = model_response

    # Helper function that strips a prompt from a model's response
    def strip_prompt_from_generated_text(response, prompt):
        return response[len(prompt):]

    # These are models that we know include prompts in their generated responses
    if (model_name == "lmsys/vicuna-7b-v1.5") or (model_name == "NousResearch/Llama-2-7b-chat-hf") or (model_name == "NousResearch/Llama-2-7b-hf") or (model_name == "instructlab/granite-7b-lab") or (model_name == "instructlab/merlinite-7b-lab"):
        model_response = strip_prompt_from_generated_text(model_response, input_prompt)

    return model_response, model_response_before_strip


def inference_qlora_ift_model(model_dir, prompt, max_new_toks=250):
    """
        Allows us to run inference on a qlora/peft ift model. 

        Parameters:
            model_dir: name of the directory where the model is located
            prompt: prompt to ask the model

        Returns:
            Model's response
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True)
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_toks, do_sample=True)
    # outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_toks, do_sample=True, top_p=0.9,temperature=0.9)
    outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    return outputs


# OLD HF INFERENCE CODE THAT WORKED ON A100
# ## code that works on a100 ###
# # Inference pipeline for llama 2 chat
# model = "NousResearch/Llama-2-7b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model)

# # model = AutoModelForCausalLM.from_pretrained(model)

# model_pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer = tokenizer,
#     torch_dtype=torch.float16,
#     max_new_tokens = 512,
#     device_map="auto",
# )

# def get_response(prompt: str) -> None:
#     """
#     Generate a response from the Llama model.

#     Parameters:
#         prompt (str): The user's input/question for the model.

#     Returns:
#         None: Prints the model's response.
#     """
#     sequences = model_pipeline(
#         prompt,
#         do_sample=True
#     )
#     # sequences = model_pipeline(
#     #     prompt,
#     #     do_sample=True,
#     #     top_k=10,
#     #     num_return_sequences=1,
#     #     eos_token_id=tokenizer.eos_token_id,
#     #     max_length=256,
#     # )
#     print("Chatbot:", sequences[0]['generated_text'])
#     return

# prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
# get_response(prompt)

# ##########



### OLD HF INFERENCE CODE THAT WORKED ON MAC
# sequences = pipeline(
#     "I'm looking to make a lemon cake. Could you please give me a recipe for a good lemon cake?",
#     do_sample=True,
# )

# print(sequences[0].get("generated_text"))

# # Inference pipeline for llama 2 no chat
# model = "NousResearch/Llama-2-7b-hf"

# tokenizer = AutoTokenizer.from_pretrained(model)

# model = AutoModelForCausalLM.from_pretrained(model)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer = tokenizer,
#     torch_dtype=torch.float16,
#     max_new_tokens = 512,
#     device_map="auto",
# )

# sequences = pipeline(
#     "I'm looking to make a lemon cake. Could you please give me a recipe for a good lemon cake?",
#     do_sample=True,
# )

# print(sequences[0].get("generated_text"))
    
