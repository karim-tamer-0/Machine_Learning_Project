"""Stateless feature builder for the classification pipeline (Member B).
 
It applies fixed row-level rules and learns nothing from the data, so it cannot leak
information between train, validation and test. It runs as the first pipeline step.
 
    Company (ID)  -> has_company (1 if a company is attached, else 0); the ID is dropped
    total_nights  =  StaysInWeekendNights + StaysInWeekNights
    total_guests  =  Adults + Children + Babies
"""
from sklearn.base import BaseEstimator, TransformerMixin
 
 
class BookingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X = X.copy()
        if "Company" in X.columns:
            X["has_company"] = (X["Company"] != "NoCompany").astype(int)
            X = X.drop(columns="Company")
        if {"StaysInWeekendNights", "StaysInWeekNights"} <= set(X.columns):
            X["total_nights"] = X["StaysInWeekendNights"] + X["StaysInWeekNights"]
        if {"Adults", "Children", "Babies"} <= set(X.columns):
            X["total_guests"] = X["Adults"] + X["Children"] + X["Babies"]
        return X