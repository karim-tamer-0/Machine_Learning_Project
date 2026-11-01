import matplotlib.pyplot as matty
import pandas as panda
import numpy as nabawy
from sklearn.preprocessing import StandardScaler
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

cc = panda.read_csv(os.getenv("H2_original_dataset")) 

# Month sine and cosine
month_number = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

month = cc['ArrivalDateMonth'].map(month_number)

cc['Month_sin'] = nabawy.sin(2 * nabawy.pi * (month - 1) / 12)
cc['Month_cos'] = nabawy.cos(2 * nabawy.pi * (month - 1) / 12)


# one hot encoding for categorical features
cc = panda.get_dummies(
    cc,
    columns=['Meal', 'MarketSegment', 'DistributionChannel', 'ReservedRoomType', 'CustomerType'],
    dtype=int
)

cc.columns = cc.columns.str.strip()


# country grouping
country_counts = cc['Country'].value_counts(dropna=False)
top_countries = country_counts.head(15).index.tolist()

cc['Country'] = cc['Country'].apply(
    lambda country: country if country in top_countries else 'Other'
)

cc = panda.get_dummies(cc, columns=['Country'], dtype=int)


# selecting features
numerical_features = [
    'LeadTime',
    'Adults',
    'Children',
    'Babies',
    'TotalOfSpecialRequests',
    'StaysInWeekendNights',
    'StaysInWeekNights',
    'ArrivalDateDayOfMonth'
]

date_features = [
    'Month_sin',
    'Month_cos'
]

encoded_features = [
    col for col in cc.columns
    if col.startswith((
        'Meal_',
        'Country_',
        'MarketSegment_',
        'DistributionChannel_',
        'ReservedRoomType_',
        'CustomerType_'
    ))
]

selected_features = numerical_features + date_features + encoded_features

kmeans_data = cc[selected_features].copy()


# handling missing values
missing = kmeans_data.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

missing_children = kmeans_data[kmeans_data['Children'].isna()]

kmeans_data['Children'] = kmeans_data['Children'].fillna(0)


# removing bookings with 0 adults
kmeans_data = kmeans_data[kmeans_data['Adults'] > 0].copy()


# adding ArrivalDateDayOfMonth
kmeans_data['ArrivalDateDayOfMonth'] = cc.loc[
    kmeans_data.index, 'ArrivalDateDayOfMonth'
]


# removing extreme values in Babies
kmeans_data = kmeans_data[kmeans_data['Babies'] <= 2].copy()


# scaling
kmeans_scaled = kmeans_data.copy()

scaler = StandardScaler()

kmeans_scaled[numerical_features] = scaler.fit_transform(
    kmeans_scaled[numerical_features]
)


# checking the final dataset
print("Final dataset shape:", kmeans_scaled.shape)

print("\nMissing values:")
print(kmeans_scaled.isnull().sum().sum())

print("\nNumerical features:")
print(kmeans_scaled[numerical_features].describe())

# saving the dataset
output_path = os.getenv("output_clustering_processing")

kmeans_scaled.to_csv(output_path + r"\kmeans_cleaned_scaled.csv", index=False)

joblib.dump(scaler, output_path + r"\kmeans_scaler.pkl")