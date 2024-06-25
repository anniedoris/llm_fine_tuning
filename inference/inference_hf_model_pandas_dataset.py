# import sys
# sys.path.append("utils")
from inference import *
import pandas as pd
from tqdm import tqdm
import os

### DIFFERENT MODEL OPTIONS FOR HF INFERENCE
### "NousResearch/Llama-2-7b-chat-hf"
### "NousResearch/Llama-2-7b-hf"
### "lmsys/vicuna-7b-v1.5"
### "instructlab/granite-7b-lab"
### "instructlab/merlinite-7b-lab"

inference_save_dir_parent = "design_qa"

# TODO: figure out why lmsys/vicuna doesn't return anything?
model_name = "instructlab/granite-7b-lab"
question_df = pd.read_csv('design_qa/context_and_prompts.csv')
response_df = pd.DataFrame(columns=['question', 'model_prediction', 'ground_truth', 'complete_response'])

for i, row in tqdm(question_df.iterrows(), total=len(question_df), desc='generating model responses for retrieval qa'):
    if i in [0, 1, 2]:
        question = row['prompt_with_context']
        model_response, complete_response = inference_hf_model(model_name, question)
        # print("RESPONSE:")
        # print(model_response)
        # print("COMPLETE RESPONSE")
        # print(complete_response)
        row = {'question': question, 'ground_truth': row['ground_truth'], 'model_prediction': model_response, 'complete_response': complete_response}
        response_df = pd.concat([response_df, pd.DataFrame([row])], ignore_index = True)

# Save the results from the inference into the designqa folder
model_save_dir = model_name.replace('/', "_")
if model_save_dir in os.listdir(inference_save_dir_parent):
    pass
else:
    os.mkdir(inference_save_dir_parent + '/' + model_save_dir)
response_df.to_csv(inference_save_dir_parent + '/' + model_save_dir + "/" + model_save_dir + '_retrievalRAG.csv')