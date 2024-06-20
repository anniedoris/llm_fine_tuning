from datasets import load_dataset
import sys
sys.path.append("design_qa")

dataset = load_dataset('csv', data_files='design_qa/context_and_prompts.csv')
dataset.push_to_hub("designqa_retrieval_with_context", private=True)