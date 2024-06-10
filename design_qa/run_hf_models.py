from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
sys.path.append("utils")
from inference import *
import transformers
import torch
from torch import cuda
from transformers import GenerationConfig

# Referenced this video and associated colab: https://www.youtube.com/watch?v=Z6sCl6abJj4

def inference_hf_model(model_name, input_prompt, max_new_toks=520):
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

    # TODO: play with other parameters settings for inference?

#     # sequences = model_pipeline(
#     #     prompt,
#     #     do_sample=True,
#     #     top_k=10,
#     #     num_return_sequences=1,
#     #     eos_token_id=tokenizer.eos_token_id,
#     #     max_length=256,
#     # )

    model_response = sequences[0]['generated_text']
    #TODO: for llama-2-7b-chat and llama-2-7b-hf, prompt is included in the response. Need to remove this

    def strip_prompt_from_generated_text(response, prompt):
        print(len(strip_prompt_from_generated_text))
        return

    print("REMOVING PROMPT")
    strip_prompt_from_generated_text(model_response, input_prompt)
    return model_response

# NousResearch/Llama-2-7b-chat-hf
# "NousResearch/Llama-2-7b-hf"
model_response = inference_hf_model("lmsys/vicuna-7b-v1.5", 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n')
print("MODEL:")
print(model_response)


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