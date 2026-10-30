import pandas as pd

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