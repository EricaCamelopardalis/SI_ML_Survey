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
 
 
def prepare_model_data_tfidf(t_col, r_col, Survey_df, Likert_Guide_df, notebook_path, vectorizer=None):
    agreement_idx = get_agreement_index(r_col, Survey_df, Likert_Guide_df)
    subset = Survey_df.loc[agreement_idx].copy()
    subset = subset.dropna(subset=[r_col])
    subset = subset[subset[t_col + "_lemma_str"].str.strip() != ""]
    subset = subset[subset[r_col] != 3]
    y = (subset[r_col] > 3).astype(int)
    if len(y) < 50 or y.nunique() < 2:
        return None, None, None
    # Use pre-loaded vectorizer if provided, otherwise load from disk.
    if vectorizer is None:
        vectorizer = load_tfidf_vectorizer(t_col, notebook_path)
    X = vectorizer.transform(subset[t_col + "_lemma_str"].tolist())
    return X, y, vectorizer

def evaluate_model(model, vocab, X_test, y_train, y_test,
                   vocab_coverage, t_col, r_col, label, y):
    # Compute all classification metrics and extract top features.
    # Works for any model with predict() and predict_proba() methods.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    pos_pct = round(y.mean() * 100, 1)
    neg_pct = round(100 - pos_pct, 1)
    return {
        "Label": label,
        "T_Col": t_col,
        "R_Col": r_col,
        "N_Train": len(y_train),
        "N_Test": len(y_test),
        "Pos_Pct": pos_pct,
        "Neg_Pct": neg_pct,
        "Vocab_Size": len(vocab) if vocab is not None else None,
        "Vocab_Coverage": vocab_coverage,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "F1": round(f1_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob), 3),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 3),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Top_Features": get_top_features_by_ratio(model, vocab) if vocab is not None else None,
    }

# Feature importance.

def get_top_features_by_ratio(nb_model, vocab, n=20):
    # Extract top n tokens most distinctive to each class using likelihood ratios.
    # Heck yeah information criterions. 
    # More informative than raw log probabilities which favor common tokens.
    log_prob_neg = nb_model.feature_log_prob_[0]
    log_prob_pos = nb_model.feature_log_prob_[1]
    neg_ratio = log_prob_neg - log_prob_pos
    pos_ratio = log_prob_pos - log_prob_neg
    top_neg_idx = neg_ratio.argsort()[-n:][::-1]
    top_pos_idx = pos_ratio.argsort()[-n:][::-1]
    return {
        0: [vocab[i] for i in top_neg_idx],
        1: [vocab[i] for i in top_pos_idx],
    }

# Results formatting

def build_comparison_df(Results_By_Technique):
    # Flatten Results_By_Technique into a single comparison DataFrame.
    # Top_Features excluded.
    import pandas as pd
    rows = []
    for technique_name, results in Results_By_Technique.items():
        for r in results:
            rows.append({
                "Technique": technique_name,
                "Pairing": r["Label"],
                "N": r["N_Train"] + r["N_Test"],
                "Vocab Size": r["Vocab_Size"],
                "Pos %": r["Pos_Pct"],
                "Neg %": r["Neg_Pct"],
                "Accuracy": r["Accuracy"],
                "F1": r["F1"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "ROC AUC": r["ROC_AUC"],
                "MCC": r["MCC"],
                "Vocab Coverage": r["Vocab_Coverage"],
            })
    return pd.DataFrame(rows)