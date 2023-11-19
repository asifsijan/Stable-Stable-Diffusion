

import pandas as pd






file_name = "name.csv"

df = pd.read_csv(file_name)

temp = df.head()

# Set max_colwidth to None to display the full content of each column
pd.set_option('display.max_colwidth', None)

for index, row in temp.iterrows():
    print(row.to_string(index=False))





file_name = 'beluga_desc.csv'



#file_name = "description.csv"

df = pd.read_csv(file_name)

temp = df.head()

# Set max_colwidth to None to display the full content of each column
pd.set_option('display.max_colwidth', None)

for index, row in temp.iterrows():
	print(row.to_string(index=False))
	print()
	print()

