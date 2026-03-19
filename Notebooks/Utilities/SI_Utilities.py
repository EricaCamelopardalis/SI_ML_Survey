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

# Configurations.

Leader_R_Cols = [
    "R_Leader_Helps_Understanding_encoded",
    "R_Leader_Subject_Competence_encoded",
    "R_Leader_Has_Plan_encoded",
    "R_Leader_Willing_To_Help_encoded",
]
 
Primary_Pairs = [
    ("T_Collaboration_Understanding", "R_Collaborative_Activities_encoded"),
    ("T_Leader_Performance_Suggestions", "R_Leader_Helps_Understanding_encoded"),
    ("T_Leader_Performance_Suggestions", "R_Leader_Subject_Competence_encoded"),
    ("T_Leader_Performance_Suggestions", "R_Leader_Has_Plan_encoded"),
    ("T_Leader_Performance_Suggestions", "R_Leader_Willing_To_Help_encoded"),
    ("T_Leader_Performance_Suggestions", "R_Leader_Average_encoded"),
    ("T_Other_Suggestions", "R_Overall_Session_Helpfulness_encoded"),
    ("T_Program_Overall_Suggestions", "R_Overall_Session_Helpfulness_encoded"),
]
 
# Likert scale encodings for reference.
agreement_scale = {
    "Strongly agree": 5,
    "Moderately agree": 4,
    "Neither disagree nor agree": 3,
    "Moderately disagree": 2,
    "Strongly disagree": 1,
    "Unable to judge": None,
}
 
helpfulness_scale = {
    "Extremely helpful": 5,
    "Very helpful": 4,
    "Moderately helpful": 3,
    "Slightly helpful": 2,
    "Not at all helpful": 1,
    "Unable to judge": None,
}
 
rating_encoding = {**agreement_scale, **helpfulness_scale}

# Agreement scale filtering
 
def get_agreement_index(r_col, Survey_df, Likert_Guide_df):
    # Return row indices from Survey_df that used the Agreement scale
    # for the given R_ column.
    col_name = r_col.replace("_encoded", "")
 
    if col_name == "R_Leader_Average":
        # Intersect Agreement rows across all four leader columns.
        indices = None
        for leader_col in Leader_R_Cols:
            agreement_rows = Likert_Guide_df[
                (Likert_Guide_df["Column"] == leader_col.replace("_encoded", "")) &
                (Likert_Guide_df["Scale"] == "Agreement")
            ][["Discipline", "Course_Code", "Semester", "Year"]]
            mask = Survey_df[["Discipline", "Course_Code", "Semester", "Year"]].merge(
                agreement_rows, how="inner"
            ).index
            indices = mask if indices is None else indices.intersection(mask)
        return indices
 
    agreement_rows = Likert_Guide_df[
        (Likert_Guide_df["Column"] == col_name) &
        (Likert_Guide_df["Scale"] == "Agreement")
    ][["Discipline", "Course_Code", "Semester", "Year"]]
 
    mask = Survey_df.merge(
        agreement_rows,
        on=["Discipline", "Course_Code", "Semester", "Year"],
        how="inner"
    ).index
    return mask

# Feature construction.
# BOW.
 
def build_bow(token_lists):
    # Build a sparse CSR bag-of-words matrix from a list of token lists.
    # Returns (matrix, vocab) where vocab is a sorted list of unique tokens.
    vocab = sorted(set(tok for tokens in token_lists for tok in tokens))
    vocab_index = {tok: i for i, tok in enumerate(vocab)}
    # BOW is ~99% zeros: sparse storage avoids memory issues.
    # Learned this one the hard way, funnily enough. 
    matrix = lil_matrix((len(token_lists), len(vocab)), dtype=np.int32)
    for i, tokens in enumerate(token_lists):
        counts = Counter(tokens)
        for tok, count in counts.items():
            if tok in vocab_index:
                matrix[i, vocab_index[tok]] = count
    return csr_matrix(matrix), vocab

# Data preparation.
def prepare_model_data(t_col, r_col, Survey_df, Likert_Guide_df):
    # Filter to Agreement-scale rows, build BOW features, binarize labels.
    # Returns (subset, y, features) or (None, None, None) if insufficient data.
    agreement_idx = get_agreement_index(r_col, Survey_df, Likert_Guide_df)
    subset = Survey_df.loc[agreement_idx].copy()
 
    # Combine lemmas and bigrams element-wise: cast to list to avoid
    # numpy array concatenation issues after index filtering.
    subset["features"] = subset.apply(
        lambda row: list(row[t_col + "_lemmas"]) + list(row[t_col + "_bigrams"]),
        axis=1
    )
 
    # Drop rows with no text, no rating, or neutral rating.
    # Threes mean "neither agree nor disagree": ambiguous for binary classification.
    subset = subset.dropna(subset=[r_col])
    subset = subset[subset["features"].apply(len) > 0]
    subset = subset[subset[r_col] != 3]
 
    # Binarize: above 3 = positive (1), below 3 = negative (0).
    y = (subset[r_col] > 3).astype(int)
 
    if len(y) < 50 or y.nunique() < 2:
        return None, None, None
 
    return subset, y, subset["features"].tolist()
 
 
def prepare_model_data_tfidf(t_col, r_col, Survey_df, Likert_Guide_df, notebook_path):
    # Filter to Agreement-scale rows, build TF-IDF features, binarize labels.
    # Loads the fitted vectorizer from disk for the given T_ column.
    # Returns (X, y, vectorizer) or (None, None, None) if insufficient data.
    agreement_idx = get_agreement_index(r_col, Survey_df, Likert_Guide_df)
    subset = Survey_df.loc[agreement_idx].copy()
 
    # Drop rows with no text, no rating, or neutral rating.
    subset = subset.dropna(subset=[r_col])
    subset = subset[subset[t_col + "_lemma_str"].str.strip() != ""]
    subset = subset[subset[r_col] != 3]
 
    # Binarize: above 3 = positive (1), below 3 = negative (0).
    y = (subset[r_col] > 3).astype(int)
 
    if len(y) < 50 or y.nunique() < 2:
        return None, None, None
 
    # Load fitted vectorizer and transform lemma strings.
    vectorizer = load_tfidf_vectorizer(t_col, notebook_path)
    X = vectorizer.transform(subset[t_col + "_lemma_str"].tolist())
 
    return X, y, vectorizer