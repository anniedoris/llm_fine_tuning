from metrics import eval_retrieval_qa

csv_name = 'lmsys_vicuna-7b-v1.5_retrievalRAG.csv'
av, all_scores = eval_retrieval_qa(csv_name)

with open(csv_name.strip('.csv') + ".txt", 'w') as file:
    file.write(f"Macro avg: {av}")
    file.write(f"\nAll scores: {all_scores}")

