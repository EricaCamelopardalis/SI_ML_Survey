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
 