import sys
sys.path.append("utils")
from inference import *
import pandas as pd
from tqdm import tqdm

### DIFFERENT MODEL OPTIONS FOR HF INFERENCE
### "NousResearch/Llama-2-7b-chat-hf"
### "NousResearch/Llama-2-7b-hf"
### "lmsys/vicuna-7b-v1.5"

model_name = "lmsys/vicuna-7b-v1.5"
question_df = pd.read_csv('design_qa/context_and_prompts.csv')
response_df = pd.DataFrame(columns=['question', 'model_prediction', 'ground_truth'])

for i, row in tqdm(question_df.iterrows(), total=len(question_df), desc='generating model responses for retrieval qa'):
    question = row['prompt_with_context']
    model_response, complete_response = inference_hf_model(model_name, question)
    # print("RESPONSE:")
    # print(model_response)
    # print("COMPLETE RESPONSE")
    # print(complete_response)
    row = {'question': question, 'ground_truth': row['ground_truth'], 'model_prediction': model_response}
    response_df = pd.concat([response_df, pd.DataFrame([row])], ignore_index = True)
response_df.to_csv(model_name.replace('/', "_") + '_retrievalRAG.csv')


# prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
# model_response = inference_hf_model("NousResearch/Llama-2-7b-chat-hf", prompt)
# print("MODEL:")
# print(model_response)



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