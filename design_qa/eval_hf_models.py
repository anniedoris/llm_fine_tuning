import sys
sys.path.append("utils")
from inference import *

### DIFFERENT MODEL OPTIONS FOR HF INFERENCE
### "NousResearch/Llama-2-7b-chat-hf"
### "NousResearch/Llama-2-7b-hf"
### "lmsys/vicuna-7b-v1.5"

prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
model_response = inference_hf_model("NousResearch/Llama-2-7b-chat-hf", prompt)
print("MODEL:")
print(model_response)

### FOR PEFT IFT MODEL INFERENCE
# prompt = f"""### Instruction:
# Use the Input below to create an instruction, which could have been used to generate the input using an LLM.
 
# ### Input:
# 1 cup of sugar, 2 cups of flour, lots of lemon juice and lemon zest, baking soda, salt, and vanilla extract.
 
# ### Response:
# """
# model_response = inference_qlora_ift_model("llama-7-int4-dolly", prompt)
# print("IFT MODEL:")
# print(model_response)