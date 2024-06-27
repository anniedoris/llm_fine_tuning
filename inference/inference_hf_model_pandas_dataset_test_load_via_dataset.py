from inference import *
import pandas as pd
from tqdm import tqdm
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import time

### DIFFERENT MODEL OPTIONS FOR HF INFERENCE
### "NousResearch/Llama-2-7b-chat-hf"
### "NousResearch/Llama-2-7b-hf"
### "lmsys/vicuna-7b-v1.5"
### "instructlab/granite-7b-lab"
### "instructlab/merlinite-7b-lab"

inference_save_dir_parent = "design_qa"

# Load dataset from the hub
dataset = load_dataset("anniedoris/designqa_retrieval_idealrag", split='train')

# TODO: figure out why lmsys/vicuna doesn't return anything?
model_name = "NousResearch/Llama-2-7b-chat-hf"

# TODO: should I be using Llamatokenizer?
tokenizer = AutoTokenizer.from_pretrained(model_name)

start_time = time.time()

print("Starting model loading...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Ending model loading...")

model_pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer = tokenizer,
    torch_dtype=torch.float16,
    max_new_tokens = 250,
    device_map="auto"
    )

all_responses = []
counter = 0
for out in model_pipeline(KeyDataset(dataset, "question")):
    text_response = out[0]['generated_text']
    print(counter)
    counter += 1
    all_responses.append(text_response)
    # print(out)

response_df = pd.DataFrame({'model_prediction': all_responses})
response_df.to_csv('test_optimize.csv')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# question_df = pd.read_csv('design_qa/datasets/ideal_rag.csv')
# response_df = pd.DataFrame(columns=['question', 'model_prediction', 'ground_truth', 'complete_response'])

# # for i, row in tqdm(question_df.iterrows(), total=len(question_df), desc='generating model responses for retrieval qa'):
#     # question = row['question']

# model_responses, complete_responses = inference_hf_model(model_name, question_df['question'].tolist())

# inference_data = {
#     'question': question_df['question'].tolist(),
#     'model_prediction': model_responses,
#     'ground_truth': question_df['ground_truth'].tolist(),
#     'complete_response': complete_responses
# }

# response_df = pd.DataFrame(inference_data)

#     # row = {'question': question, 'ground_truth': row['ground_truth'], 'model_prediction': model_response, 'complete_response': complete_response}
#     # response_df = pd.concat([response_df, pd.DataFrame([row])], ignore_index = True)

# # Save the results from the inference into the designqa folder
# model_save_dir = model_name.replace('/', "_")
# if model_save_dir in os.listdir(inference_save_dir_parent):
#     pass
# else:
#     os.mkdir(inference_save_dir_parent + '/' + model_save_dir)
# response_df.to_csv(inference_save_dir_parent + '/' + model_save_dir + "/" + model_save_dir + '_test5.csv')