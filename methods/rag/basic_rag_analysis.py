import pandas as pd

df = pd.read_csv('design_qa/context_and_prompts.csv')

count = 0
for i, row in df.iterrows():
    gt = row["ground_truth"]
    if gt in row["prompt_with_context"]:
        count +=1
print("Result")
print(count)
    # print(row['prompt_without_context'].split("does rule ")[1].split(" state exactly")[0])