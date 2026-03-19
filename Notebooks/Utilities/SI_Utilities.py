# NLP Utilities for SI Data.
# Created by Clara Smith
# Orderly management of modeling utility functions.
# Import this module at the top of any modeling notebook.

# Scientific computing.
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix
 
# Machine learning.
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTETomek
 
# File handling.
from pathlib import Path
import json
import pickle
 
 # Data Access Helpers

def load_survey_data(notebook_path):
    # Load Survey_df, Likert_Guide_df, and Column_Metadata from standard paths.
    # Pass Path().resolve() from the calling notebook as notebook_path.
    project_root = Path(notebook_path).parents[1]
    data_dir = project_root / "Clean_Data_Resources"
 
    Survey_df = pd.read_parquet(data_dir / "Survey_df_Text_Parsed.parquet")
    Likert_Guide_df = pd.read_csv(data_dir / "Survey_Results_Likert_Guide.csv")
 
    with open(data_dir / "column_metadata.json", "r") as f:
        Column_Metadata = json.load(f)
 
    return Survey_df, Likert_Guide_df, Column_Metadata
 
 
def load_tfidf_vectorizer(t_col, notebook_path):
    # Load a fitted TfidfVectorizer for a given T_ column.
    project_root = Path(notebook_path).parents[1]
    vectorizer_dir = project_root / "Clean_Data_Resources" / "Vectorizers"
    vectorizer_path = vectorizer_dir / f"tfidf_vectorizer_{t_col}.pkl"
    with open(vectorizer_path, "rb") as f:
        return pickle.load(f)
