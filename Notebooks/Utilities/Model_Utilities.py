# Model_Utilities.py
# Shared modeling functions for SI Survey NLP notebooks.
# Import this module at the top of any modeling notebook.
 
# Scientific computing.
import numpy as np
 
# Machine learning.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Utilities imports.
from SI_Utilities import prepare_model_data_tfidf, evaluate_model
 
# Local utilities.
import sys
from pathlib import Path

# Naive Bayes with TF-IDF
 
def split_and_train_tfidf_nb(X, y):
    # 80/20 stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026, stratify=y
    )
    # SMOTE on training set only: never resample the test set.
    smote = SMOTE(random_state=2026)
    X_train_r, y_train_r = smote.fit_resample(X_train, y_train)
    nb = MultinomialNB()
    nb.fit(X_train_r, y_train_r)
    return nb, X_test, y_train_r, y_test
 
 
def run_tfidf_nb(t_col, r_col, label, Survey_df, Likert_Guide_df, Notebook_Dir, vectorizer=None):
    # Prepare TF-IDF features and binarized labels for a given T/R pairing.
    # Returns None if data is insufficient or only one class is present.
    X, y, vectorizer = prepare_model_data_tfidf(
        t_col, r_col, Survey_df, Likert_Guide_df, Notebook_Dir, vectorizer
    )
    if X is None:
        print(f"Skipping {label}.")
        return None
    nb, X_test, y_train, y_test = split_and_train_tfidf_nb(X, y)
    result = evaluate_model(
        nb, None, X_test, y_train, y_test, None, t_col, r_col, label, y
    )
    # Store vectorizer for feature extraction later.
    result["Vectorizer"] = vectorizer
    return result