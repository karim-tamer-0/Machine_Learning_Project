import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =====================================================================
# 1. LOAD DATA FROM `All_models_data.csv`
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

input_path = os.path.join(project_root, "src", "data", "All_models_data.csv")
output_dir = os.path.join(project_root, "src", "data", "regression_data")
artifacts_dir = os.path.join(project_root, "artifacts")

# If `regression_data` or `artifacts` is an empty placeholder file, turn it into a folder
for folder in [output_dir, artifacts_dir]:
    if os.path.isfile(folder):
        os.remove(folder)
    os.makedirs(folder, exist_ok=True)

df = pd.read_csv(input_path)
print("1. Initial shape from All_models_data.csv:", df.shape)

# =====================================================================
# 2. CONVERT "NULL" TEXT STRINGS & CREATE AGENT/COMPANY FLAGS
# =====================================================================
# In H2.csv, missing Agent/Company/Country values are written as the word "NULL"
df = df.replace(["NULL", " NULL", "nan"], pd.NA)

# Create simple 0/1 flags for whether an Agent or Company was used
df["HasAgent"] = df["Agent"].notna().astype(int)
df["HasCompany"] = df["Company"].notna().astype(int)

# =====================================================================
# 3. DROP LEAKAGE COLUMNS & RAW ID COLUMNS
# =====================================================================
# We predict ADR at initial booking creation. Anything that happens LATER is leakage.
columns_to_drop = [
    "IsCanceled",             # Leakage: Future cancellation outcome (Task B target)
    "ReservationStatus",      # Leakage: Final checkout/cancel status
    "ReservationStatusDate",  # Leakage: Date of final status
    "AssignedRoomType",       # Leakage: Assigned at check-in (ReservedRoomType is kept)
    "BookingChanges",         # Leakage: Changes made after booking was created
    "DaysInWaitingList",      # Leakage: Waiting time measured after booking entry
    "Agent",                  # Replaced by HasAgent (0/1)
    "Company",                # Replaced by HasCompany (0/1)
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print("2. Shape after dropping leakage & ID columns:", df.shape)

# =====================================================================
# 4. DROP EXTREME OUTLIER & DUPLICATE ROWS
# =====================================================================
# H2.csv has 1 corrupted row where ADR = 5400, which ruins Linear Regression
df = df[df["ADR"] <= 1000]

# Remove exact duplicate rows so identical rows don't appear in both Train and Test
df = df.drop_duplicates().reset_index(drop=True)
print("3. Shape after removing ADR > 1000 outlier and duplicates:", df.shape)

# =====================================================================
# 5. SIMPLE FEATURE PREPARATION
# =====================================================================
# Convert ArrivalDateMonth from text ("January".."December") to numbers (1..12)
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
df["ArrivalDateMonth"] = df["ArrivalDateMonth"].map(month_map)

# Fill any remaining missing Country values with the mode
df["Country"] = df["Country"].fillna(df["Country"].mode()[0])

# Save the clean, readable dataset before splitting/scaling
df.to_csv(os.path.join(output_dir, "regression_cleaned.csv"), index=False)

# =====================================================================
# 6. SPLIT INTO TRAIN (70%), VALIDATION (15%), AND TEST (15%)
# =====================================================================
# MUST split BEFORE fitting encoders or scalers to prevent Data Leakage!
X = df.drop(columns=["ADR"])
y = df["ADR"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

# Keep only Top 10 countries (learned from X_train ONLY!) and group the rest as "Other"
top_10_countries = X_train["Country"].value_counts().nlargest(10).index
X_train["Country"] = X_train["Country"].where(X_train["Country"].isin(top_10_countries), "Other")
X_val["Country"] = X_val["Country"].where(X_val["Country"].isin(top_10_countries), "Other")
X_test["Country"] = X_test["Country"].where(X_test["Country"].isin(top_10_countries), "Other")

# =====================================================================
# 7. ONE-HOT ENCODE CATEGORICALS & SCALE NUMERICALS (FIT ON TRAIN ONLY)
# =====================================================================
categorical_cols = [
    "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType"
]
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

# Fit OneHotEncoder on X_train ONLY (drop='first' prevents multicollinearity in Linear Regression)
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoder.fit(X_train[categorical_cols])

# Fit StandardScaler on X_train ONLY
scaler = StandardScaler()
scaler.fit(X_train[numerical_cols])

def apply_preprocessing(X_split, y_split):
    # Scale numerical columns
    num_df = pd.DataFrame(
        scaler.transform(X_split[numerical_cols]),
        columns=numerical_cols,
        index=X_split.index
    )
    # Encode categorical columns
    cat_df = pd.DataFrame(
        encoder.transform(X_split[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_split.index
    )
    # Combine back together with target ADR
    final_df = pd.concat([num_df, cat_df], axis=1)
    final_df["ADR"] = y_split.values
    return final_df

train_final = apply_preprocessing(X_train, y_train)
val_final = apply_preprocessing(X_val, y_val)
test_final = apply_preprocessing(X_test, y_test)

# =====================================================================
# 8. SAVE CSVs TO `src/data/regression_data` & ARTIFACT TO `artifacts`
# =====================================================================
train_final.to_csv(os.path.join(output_dir, "train_regression.csv"), index=False)
val_final.to_csv(os.path.join(output_dir, "val_regression.csv"), index=False)
test_final.to_csv(os.path.join(output_dir, "test_regression.csv"), index=False)

joblib.dump(
    {"scaler": scaler, "encoder": encoder, "top_10_countries": list(top_10_countries)},
    os.path.join(artifacts_dir, "regression_preprocessor.joblib")
)

print("4. Saved preprocessed splits successfully!")
print("   Train shape:", train_final.shape)
print("   Val shape  :", val_final.shape)
print("   Test shape :", test_final.shape)

# =====================================================================
# 9. INSPECT PROCESSED DATA FOR QUALITY ASSURANCE
# =====================================================================
print("\n================ DATA QUALITY INSPECTION ================")
print(f"1. Any missing values in Train? {train_final.isnull().sum().sum() == 0}")
print(f"2. Any missing values in Val?   {val_final.isnull().sum().sum() == 0}")
print(f"3. Any missing values in Test?  {test_final.isnull().sum().sum() == 0}")
print("\n4. Target Variable (ADR) Summary on Train:")
print(train_final["ADR"].describe())
print("\n5. First 5 Features in Train:")
print(list(train_final.columns[:5]))
print("=========================================================")



