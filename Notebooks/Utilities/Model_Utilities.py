# Model_Utilities.py
# Shared modeling functions for SI Survey NLP notebooks.
# Import this module at the top of any modeling notebook.
 
# Scientific computing.
import numpy as np
 
# Machine learning.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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