# Model_Utilities.py
# Shared modeling functions for SI Survey NLP notebooks.
# Import this module at the top of any modeling notebook.
 
# Scientific computing.
import numpy as np
import pandas as pd
 
# Machine learning.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
from xgboost import XGBClassifier

# Utilities imports.
from SI_Utilities import prepare_model_data_tfidf, evaluate_model
 
# Local utilities.
from pathlib import Path

# Naive Bayes with TF-IDF
 
def split_and_train_tfidf_nb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026, stratify=y
    )
    # Invert class frequencies as priors.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    total = len(y_train)
    class_prior = [neg_count / total, pos_count / total]
    nb = MultinomialNB(class_prior=class_prior)
    nb.fit(X_train, y_train)
    return nb, X_test, y_train, y_test
 
 
def run_tfidf_nb(t_col, r_col, label, Survey_df, Likert_Guide_df, project_root, vectorizer=None):
    # Prepare TF-IDF features and binarized labels for a given T/R pairing.
    # Returns None if data is insufficient or only one class is present.
    X, y, vectorizer = prepare_model_data_tfidf(
        t_col, r_col, Survey_df, Likert_Guide_df, project_root, vectorizer
    )
    if X is None:
        print(f"Skipping {label}.")
        return None
    nb, X_test, y_train, y_test = split_and_train_tfidf_nb(X, y)
    result = evaluate_model(
        nb, None, X_test, y_train, y_test, None, t_col, r_col, label, y
    )
    # Store vectorizer and model for feature extraction and live predictions later.
    result["Vectorizer"] = vectorizer
    result["Model"] = nb
    return result

# Logistic Regression with TF-IDF.
 
def split_and_train_tfidf_lr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026, stratify=y
    )
    # class_weight handles imbalance natively: no resampling needed.
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=2026,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    return lr, X_test, y_train, y_test
 
 
def get_top_features_lr(lr_model, vectorizer, n=20):
    # Extract top n phrases most predictive of each class from LR coefficients.
    # Negative coefficients predict low ratings, positive predict high ratings.
    vocab = vectorizer.get_feature_names_out()
    coefs = lr_model.coef_[0]
    top_neg_idx = coefs.argsort()[:n]
    top_pos_idx = coefs.argsort()[-n:][::-1]
    return {
        0: [vocab[i] for i in top_neg_idx],
        1: [vocab[i] for i in top_pos_idx],
    }
 
 
def run_tfidf_lr(t_col, r_col, label, Survey_df, Likert_Guide_df, project_root, vectorizer=None):
    # Prepare TF-IDF features and binarized labels for a given T/R pairing.
    # Returns None if data is insufficient or only one class is present.
    X, y, vectorizer = prepare_model_data_tfidf(
        t_col, r_col, Survey_df, Likert_Guide_df, project_root, vectorizer
    )
    if X is None:
        print(f"Skipping {label}.")
        return None
    lr, X_test, y_train, y_test = split_and_train_tfidf_lr(X, y)
    result = evaluate_model(
        lr, None, X_test, y_train, y_test, None, t_col, r_col, label, y
    )
    result["Top_Features"] = get_top_features_lr(lr, vectorizer)
    result["Vectorizer"] = vectorizer
    result["Model"] = lr
    return result
 
 
# XGBoost with TF-IDF.
 
def split_and_train_tfidf_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026, stratify=y
    )
    # scale_pos_weight handles imbalance natively: ratio of negative to positive.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count
    xgb = XGBClassifier(
        scale_pos_weight=scale,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=2026,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    return xgb, X_test, y_train, y_test
 
 
def get_top_features_xgb(xgb_model, vectorizer, X_test, n=20):
    # Extract top n phrases most predictive of each class using SHAP values.
    vocab = vectorizer.get_feature_names_out()
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    # Positive mean SHAP predicts positive class, negative predicts negative class.
    mean_shap_signed = shap_values.mean(axis=0)
    top_neg_idx = mean_shap_signed.argsort()[:n]
    top_pos_idx = mean_shap_signed.argsort()[-n:][::-1]
    return {
        0: [vocab[i] for i in top_neg_idx],
        1: [vocab[i] for i in top_pos_idx],
        "shap_values": shap_values,
        "mean_shap": np.abs(shap_values).mean(axis=0),
    }
 
 
def run_tfidf_xgb(t_col, r_col, label, Survey_df, Likert_Guide_df, project_root, vectorizer=None):
    # Prepare TF-IDF features and binarized labels for a given T/R pairing.
    # Returns None if data is insufficient or only one class is present.
    X, y, vectorizer = prepare_model_data_tfidf(
        t_col, r_col, Survey_df, Likert_Guide_df, project_root, vectorizer
    )
    if X is None:
        print(f"Skipping {label}.")
        return None
    xgb, X_test, y_train, y_test = split_and_train_tfidf_xgb(X, y)
    result = evaluate_model(
        xgb, None, X_test, y_train, y_test, None, t_col, r_col, label, y
    )
    result["Top_Features"] = get_top_features_xgb(xgb, vectorizer, X_test)
    result["Vectorizer"] = vectorizer
    result["Model"] = xgb
    return result
 
 
# Results formatting.
 
def build_summary_tfidf_df(results):
    # Flatten a list of model result dicts into a summary DataFrame.
    # Vectorizer, Model, and Top_Features excluded: inspect separately.
    return pd.DataFrame([{
        "Pairing": r["Label"],
        "N": r["N_Train"] + r["N_Test"],
        "Pos %": r["Pos_Pct"],
        "Neg %": r["Neg_Pct"],
        "Accuracy": r["Accuracy"],
        "F1": r["F1"],
        "Precision": r["Precision"],
        "Recall": r["Recall"],
        "ROC AUC": r["ROC_AUC"],
        "MCC": r["MCC"],
    } for r in results])