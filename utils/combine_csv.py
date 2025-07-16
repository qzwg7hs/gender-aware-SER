import pandas as pd

main_dir = "C:/Users/Aruay/Desktop/ra application/project/"

# Load datasets
df1 = pd.read_csv(main_dir+'dataset/featureEmodb_Ek_new.csv')
df2 = pd.read_csv(main_dir+'dataset/featureEMOVO.csv')
df3 = pd.read_csv(main_dir+'dataset/featureNormal_Ek.csv')

# Keep only the first 194 columns
df1 = df1.iloc[:, :194]
df2 = df2.iloc[:, :194]
df3 = df3.iloc[:, :194]

# Rename the last column in df3 if it's named "0.1"
if df3.columns[-1] == "0.1":
    df3.rename(columns={"0.1": "gender"}, inplace=True)

# Ensure the column name is "gender" in all datasets
df1.rename(columns={df1.columns[-1]: "gender"}, inplace=True)
df2.rename(columns={df2.columns[-1]: "gender"}, inplace=True)

df2["gender"] = df2["gender"].map({1: 1, 2: 0})
df3["gender"] = df3["gender"].map({0: 1, 1: 0})

# Combine datasets
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save to a new CSV file
combined_df.to_csv("C:/Users/Aruay/Desktop/ra application/project/dataset/combined_speech_data.csv", index=False)

print("Combined dataset saved as 'combined_speech_data.csv'")
