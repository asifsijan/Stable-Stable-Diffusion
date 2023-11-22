
import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Open the CSV file for reading
with open('datasets/name.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Use list comprehension to join all columns into a single column
    name = [','.join(row) for row in csv_reader]

# Print or use the resulting list
# print(len(single_column_list))




pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")




prompts = []
k = 0

for i in name:

    temp = "description of" + i
    messages = [
        {
            "role": "system",
            "content": "Ans in one line:",
        },
        {"role": "user", "content": temp},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    prompts.append(outputs[0]["generated_text"])
    k += 1
    print(k)
  #  print(outputs[0]["generated_text"])

                
#    if k == 3:
#       break





transposed_list = [[desc] for desc in prompts]

# Specify the CSV file name
csv_file_name = 'datasets/zephyr_desc2.csv'

# Write the transposed list to the CSV file
with open(csv_file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(transposed_list)


print(f'Data has been added to {csv_file_name}.')