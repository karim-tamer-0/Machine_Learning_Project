"""Builds the full classification pipeline (Member B).
 
    BookingFeatures -> preprocessing -> [optional PCA/LDA] -> model
 
Because every step lives inside one sklearn Pipeline, cross-validation and tuning refit
the imputer, scaler, encoder and any PCA/LDA on the training part of each fold only.
The same fitted object is later saved and used for inference.
"""
from sklearn.pipeline import Pipeline
 
from src.features.booking_features import BookingFeatures
from src.Preprocessing.Classification import build_preprocessor
 
 
def build_pipeline(model, feature_set="strict", scale=True, reducer=None, config=None):
    """model: any sklearn-style classifier.
    scale=True for scale-sensitive models, False for tree models.
    reducer: optional PCA/LDA object, inserted between preprocessing and the model."""
    steps = [
        ("features", BookingFeatures()),
        ("prep", build_preprocessor(feature_set, scale, config)),
    ]
    if reducer is not None:
        steps.append(("reducer", reducer))
    steps.append(("model", model))
    return Pipeline(steps)