import pandas as pd

# Define your column names list
column_names = ['input_text', 'output_text']

# Read the CSV file, assuming it's named 'file.csv' and located in the current directory
df = pd.read_csv(r'C:\Users\sabrym2\struct2text\datasets\triple2ref\WebNLG_triple2ref_train.csv', header=None)

# Assign column names
df.columns = column_names

# Display the DataFrame to verify the column names
print(df.head())

# If you want to save this DataFrame back to a CSV file with column names included
df.to_csv(r'C:\Users\sabrym2\struct2text\datasets\triple2ref\WebNLG_triple2ref_train.csv', index=False)
