





import csv

# Open the CSV file for reading                                                                                                                                                                             
with open('name.csv', 'r') as csv_file:
    # Create a CSV reader object                                                                                                                                                                            
    csv_reader = csv.reader(csv_file)

    # Use list comprehension to join all columns into a single column                                                                                                                                       
    name = [','.join(row) for row in csv_reader]

# Print or use the resulting list                                                                                                                                                                           
# print(len(single_column_list))



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-13B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("stabilityai/StableBeluga-13B", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
system_prompt = ""

prompts = []
k = 0

for i in name:


    message = "One line Short description of " + i
    prompt = message
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

    prompts.append(tokenizer.decode(output[0], skip_special_tokens=True))

    k += 1
    print(k)
    #if k == 2:
     #   break


transposed_list = [[desc] for desc in prompts]

# Specify the CSV file name                                                                                                                                                                                 
csv_file_name = 'beluga_desc2.csv'

# Write the transposed list to the CSV file                                                                                                                                                                 
with open(csv_file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(transposed_list)


print(f'Data has been added to {csv_file_name}.')
