from inference import *
from datasets import load_dataset
from random import randrange
import pandas as pd
from tqdm import tqdm
import os

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

model_name = "llama-2-chat-retrieval-w-context"
inference_save_dir_parent = "design_qa"

question_df = pd.read_csv('design_qa/context_and_prompts.csv')
response_df = pd.DataFrame(columns=['question', 'model_prediction', 'ground_truth'])

# Llama instruction format
def format_instruction(row):
	return f"""### Instruction:
    {row['prompt_with_context']}

    ### Response:
    """

for i, row in tqdm(question_df.iterrows(), total=len(question_df), desc='generating model responses for retrieval qa'):

    if i in [0, 1, 2]:
        print("I: " + str(i) + "\n")
        prompt = format_instruction(row)
        model_response = inference_qlora_ift_model('models/' + model_name, prompt)
        row = {'question': row['prompt_with_context'], 'ground_truth': row['ground_truth'], 'model_prediction': model_response}
        response_df = pd.concat([response_df, pd.DataFrame([row])], ignore_index = True)

# Save the results from the inference into the designqa folder
model_save_dir = model_name.replace('/', "_")
if model_save_dir in os.listdir(inference_save_dir_parent):
    pass
else:
    os.mkdir(inference_save_dir_parent + '/' + model_save_dir)
response_df.to_csv(inference_save_dir_parent + '/' + model_save_dir + "/" + model_save_dir + '_retrievalRAG.csv')