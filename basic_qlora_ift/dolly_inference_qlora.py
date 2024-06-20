import sys
sys.path.append("utils")
from inference import *
from datasets import load_dataset
from random import randrange
import pandas as pd
from tqdm import tqdm

### OLD TEST EXAMPLE THAT QLORA WORKS FOR DOLLY EXAMPLE ###
# prompt = f"""### Instruction:
# Use the Input below to create an instruction, which could have been used to generate the input using an LLM.
 
# ### Input:
# 1 cup of sugar, 2 cups of flour, lots of lemon juice and lemon zest, baking soda, salt, and vanilla extract.
 
# ### Response:
# """
# model_response = inference_qlora_ift_model("llama-7-int4-dolly", prompt)
# print("IFT MODEL:")
# print(model_response)
#########################################


### INFERENCE ON DESIGNQA

model_name = "llama-2-retrieval-w-context"

question_df = pd.read_csv('design_qa/context_and_prompts.csv')
response_df = pd.DataFrame(columns=['question', 'model_prediction', 'ground_truth'])

def format_instruction(row):
	return f"""### Instruction:
    {row['prompt_with_context']}

    ### Response:
    """

for i, row in tqdm(question_df.iterrows(), total=len(question_df), desc='generating model responses for retrieval qa'):
    prompt = format_instruction(row)
    model_response = inference_qlora_ift_model(model_name, prompt)
    row = {'question': row['prompt_with_context'], 'ground_truth': row['ground_truth'], 'model_prediction': model_response}
    response_df = pd.concat([response_df, pd.DataFrame([row])], ignore_index = True)
# response_df.to_csv('test.csv')
response_df.to_csv(model_name.replace('/', "_") + '_retrievalRAG.csv')
