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
model_name = "NousResearch/Llama-2-7b-chat-hf"
question_df = pd.read_csv('design_qa/datasets/ideal_rag.csv')
response_df = pd.DataFrame(columns=['question', 'model_prediction', 'ground_truth', 'complete_response'])

# for i, row in tqdm(question_df.iterrows(), total=len(question_df), desc='generating model responses for retrieval qa'):
    # question = row['question']

model_responses, complete_responses = inference_hf_model(model_name, question_df['question'].tolist())

inference_data = {
    'question': question_df['question'].tolist(),
    'model_prediction': model_responses,
    'ground_truth': question_df['ground_truth'].tolist(),
    'complete_response': complete_responses
}

response_df = pd.DataFrame(inference_data)

    # row = {'question': question, 'ground_truth': row['ground_truth'], 'model_prediction': model_response, 'complete_response': complete_response}
    # response_df = pd.concat([response_df, pd.DataFrame([row])], ignore_index = True)

# Save the results from the inference into the designqa folder
model_save_dir = model_name.replace('/', "_")
if model_save_dir in os.listdir(inference_save_dir_parent):
    pass
else:
    os.mkdir(inference_save_dir_parent + '/' + model_save_dir)
response_df.to_csv(inference_save_dir_parent + '/' + model_save_dir + "/" + model_save_dir + '_test5.csv')