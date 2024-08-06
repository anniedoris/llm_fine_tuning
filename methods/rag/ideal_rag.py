# This script generates "ideal" RAG framed around Llamaindex RAG

import pandas as pd
import random

df = pd.read_csv('design_qa/datasets/context_and_prompts3.csv')

length_rag = []

for i, row in df.iterrows():
    just_context = row['prompt_with_context'].split('```')[1]
    length_rag.append(len(just_context))

# Get the average character length of simple Llamaindex RAG
avg_length = round(sum(length_rag)/len(length_rag))

def extract_excerpt(file_path, search_string, excerpt_length=avg_length):
    '''
    extracts an excerpt containing the correct rule for ideal rag
    '''

    try:

        # open rule document
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Find the position of the search string in the document
        pos = content.find(search_string)
        if pos == -1:
            return "Search string not found in the document."

        # Get random number for starting point of rule within chunk
        start_pos_excerpt = random.randint(0, excerpt_length - len(search_string))

        # figure out how this corresponds to indexing into the original document
        start_pos = pos - start_pos_excerpt
        end_pos = pos + (excerpt_length - start_pos_excerpt)
        
        # handle the case where we passed the end of the document
        if end_pos > len(content):
            end_pos = len(content)

        # print(f"excerpt length: {excerpt_length}")
        # print(f"search string: {len(search_string)}")
        # print(f"random int: {start_pos_excerpt}")
        # print(f"start pos: {start_pos}")
        # print(f"end pos: {end_pos}")
        # print(f"length content: {len(content)}")
        # print(f"delta: {end_pos - start_pos}")
        
        excerpt = content[start_pos:end_pos]
        return excerpt
    
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return str(e)

# File path and looping through the search string
file_path = 'design_qa/datasets/rules_pdfplumber1.txt'

df_ideal_rag = pd.DataFrame(columns=['ideal_rag', 'question', 'ground_truth'])

for i, row in df.iterrows():
    search_string = row['ground_truth']

    first_part_question = row['prompt_with_context'].split('```')[0]
    second_part_question = row['prompt_with_context'].split('```')[2]

    # # Extract the excerpt
    excerpt = extract_excerpt(file_path, search_string)
    row = {'ideal_rag': excerpt, 'question': first_part_question + '\n```\n' + excerpt + '\n```\n' + second_part_question, 'ground_truth': search_string}
    df_ideal_rag = pd.concat([df_ideal_rag, pd.DataFrame([row])], ignore_index=True)

# Check that ideal RAG is successful - 43 cases that fail
# For the time being, just take them out (either contain midpoint of rule or they have bullet points)
# TODO: fix these issues
number_count = 0
for i, row in df_ideal_rag.iterrows():
    rule = row['ground_truth']
    excerpt = row['ideal_rag']

    pos = excerpt.find(rule)
    if pos == -1:
        print("Search string not found in the document.")
        print(i)
        print(row['question'].split('What does rule')[1].split('state exactly')[0])
        number_count += 1
        df_ideal_rag.drop(i, inplace=True)

print(number_count)

# 43 rules are excluded since they are not there verbatim -> fix these rules
df_ideal_rag.to_csv('design_qa/datasets/ideal_rag_set3.csv')


# print("\nSEARCH STRING")
# print(search_string)
# print("\nEXCERPT")
# print(excerpt)
