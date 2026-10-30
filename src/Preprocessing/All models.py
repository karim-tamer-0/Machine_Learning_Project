import pandas as pd
import os

#Load raw dataset directly from GitHub raw URL
df = pd.read_csv(
    'https://raw.githubusercontent.com/karim-tamer-0/Machine_Learning_Project/d1893f4709bbc6cdcc358064e15796f1b795262e/data/H2.csv'
)

#Fill missing children values
df['Children'] = df['Children'].fillna(0).astype(int)

#Fill missing country values
country_mode = df['Country'].mode()[0]
df['Country'] = df['Country'].fillna(country_mode)

#Filtering impossible data
df = df[df['ADR'] > 0]

df = df[(df['StaysInWeekNights'] + df['StaysInWeekendNights']) > 0]

df = df[(df['Children'] + df['Adults'] + df['Babies']) > 0]

#Verify changes
print("Dataset shape after cleaning:", df.shape)
print("Remaining NAs in 'Children':", df['Children'].isna().sum())
print("Remaining NAs in 'Country':", df['Country'].isna().sum())

#Save to src/data/All_models_data.csv
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
target_dir = os.path.join(project_root, "src", "data")
target_file = os.path.join(target_dir, "All_models_data.csv")
os.makedirs(target_dir, exist_ok=True)
df.to_csv(target_file, index=False, header=True)