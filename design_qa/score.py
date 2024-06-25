from metrics import eval_retrieval_qa

csv_name = 'design_qa/instructlab_merlinite-7b-lab_retrievalRAG.csv'
av, all_scores = eval_retrieval_qa(csv_name)

with open(csv_name.strip('.csv') + ".txt", 'w') as file:
    file.write(f"Macro avg: {av}")
    file.write(f"\nAll scores: {all_scores}")

