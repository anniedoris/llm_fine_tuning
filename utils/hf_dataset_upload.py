from datasets import load_dataset
import sys
sys.path.append("design_qa")
sys.path.append("datasets")

# Uploading a dataset to the hub
dataset = load_dataset('csv', data_files='datasets/rule_retrieval_qa.csv')
dataset.push_to_hub("designqa_retrieval")