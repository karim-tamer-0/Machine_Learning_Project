"""Train/validation/test split and cross-validation folds for classification (Member B).
 
Protocol (70% train / 20% validation / 10% test):
    - train:      models are fitted here; 5-fold CV for tuning and CV results runs here only.
    - validation: compares baseline vs tuned models, PCA/LDA variants and model families,
                  and is used to choose the deployed model.
    - test:       touched once, at the very end, for the final reported performance.
 
Order of operations (important for leakage):
    1. clean_classification() runs first: row-level rules with fixed limits only.
    2. The split happens here, once, with a fixed seed and stratification.
    3. Every fitted step (imputer, scaler, encoder, PCA, LDA, model) is fitted later,
       inside a Pipeline, on training data only.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
 
from src.data.cleaning import clean_classification
from src.data.loader import PROJECT_ROOT, get_features, load_all_models, load_config
 
 
def make_split(df, feature_cols, target, val_size, test_size, seed):
    """Stratified train/validation/test split. val_size and test_size are fractions of
    the whole dataset. Only the requested feature columns are kept, so dropped (leaky)
    columns can never reach a model."""
    X = df[feature_cols]
    y = df[target]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size),
        stratify=y_rest, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
 
 
def get_cv(config=None):
    """Stratified K-fold used for tuning and CV results on the TRAINING set only."""
    config = config or load_config()
    n_splits = config["classification"].get("split", {}).get("cv_folds", 5)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config["seed"])
 
 
def load_classification_split(feature_set="strict", config=None):
    """Load, clean, select features, and split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    config = config or load_config()
    cfg = config["classification"]
    split_cfg = cfg.get("split", {})
    cleaned, _ = clean_classification(load_all_models(config), config)
    numeric, categorical = get_features(feature_set, config)
    return make_split(
        cleaned, numeric + categorical, cfg["target"],
        split_cfg.get("val_size", 0.2), split_cfg.get("test_size", 0.1), config["seed"])
 
 
def save_split_indices(X_train, X_val, X_test, config=None):
    """Save the row ids of each part (ids refer to the cleaned data) for reproducibility."""
    config = config or load_config()
    out_dir = PROJECT_ROOT / config["classification"]["output_dir"] / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.concat([
        pd.DataFrame({"row_id": part.index, "set": name})
        for name, part in (("train", X_train), ("validation", X_val), ("test", X_test))
    ]).sort_values("row_id")
    path = out_dir / "split_indices.csv"
    ids.to_csv(path, index=False)
    return path
 
 
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_classification_split("strict")
    total = len(X_train) + len(X_val) + len(X_test)
    for name, X, y in (("train", X_train, y_train), ("validation", X_val, y_val),
                       ("test", X_test, y_test)):
        print(f"{name:<10} {X.shape}  {len(X) / total:.1%}  cancel rate {y.mean():.3f}")
    print("saved:", save_split_indices(X_train, X_val, X_test))