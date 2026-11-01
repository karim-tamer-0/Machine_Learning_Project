"""Preprocessing for the classification task (Member B).
 
build_preprocessor() returns a ColumnTransformer meant to run AFTER BookingFeatures.
Everything here that learns from data (imputer, scaler, encoder) is fitted on the
training data only, because the preprocessor lives inside the model Pipeline.
 
Two variants:
    scale=True   for Logistic Regression, KNN, SVM, MLP, PCA, LDA
                 numeric: median impute -> StandardScaler
                 skewed counts (PreviousCancellations, PreviousBookingsNotCanceled):
                 median impute -> log1p -> StandardScaler
    scale=False  for tree models (Decision Tree, Random Forest, XGBoost)
                 numeric: median impute only, no log1p and no scaling
Categorical columns (both variants): impute -> one-hot, with rare categories grouped.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
 
from src.data.loader import get_features
 
SKEWED = ["PreviousCancellations", "PreviousBookingsNotCanceled"]
 
 
def get_column_groups(feature_set="strict", config=None):
    """Return (skewed, plain_numeric, categorical) column lists as they look AFTER
    BookingFeatures has run (Company replaced by has_company, totals added)."""
    numeric, categorical = get_features(feature_set, config)
    skewed = [c for c in numeric if c in SKEWED]
    plain = [c for c in numeric if c not in SKEWED]
    if "Company" in categorical:
        categorical = [c for c in categorical if c != "Company"]
        plain.append("has_company")
    if {"StaysInWeekendNights", "StaysInWeekNights"} <= set(numeric):
        plain.append("total_nights")
    if {"Adults", "Children", "Babies"} <= set(numeric):
        plain.append("total_guests")
    return skewed, plain, categorical
 
 
def build_preprocessor(feature_set="strict", scale=True, config=None, min_frequency=0.01):
    skewed, plain, categorical = get_column_groups(feature_set, config)
 
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency,
                                 sparse_output=False)),
    ])
 
    if scale:
        skewed_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        plain_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers = [
            ("skewed", skewed_pipe, skewed),
            ("numeric", plain_pipe, plain),
            ("categorical", categorical_pipe, categorical),
        ]
    else:
        transformers = [
            ("numeric", SimpleImputer(strategy="median"), skewed + plain),
            ("categorical", categorical_pipe, categorical),
        ]
 
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)