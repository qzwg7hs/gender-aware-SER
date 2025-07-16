import pandas as pd

file_path = "C:/Users/Aruay/Desktop/ra application/project/dataset/featureNormal_Ek.csv"

df = pd.read_csv(file_path)

total_rows = len(df)

speaker_column = []

speaker_value = 1
for i in range(total_rows):
    speaker_column.append(speaker_value)
    if (i + 1) % 60 == 0:  # Increment every 98 rows, stop at 6
        speaker_value += 1

# Add the "speaker" column to the DataFrame
df['speaker'] = speaker_column

# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)

print("Speaker column added successfully!")
