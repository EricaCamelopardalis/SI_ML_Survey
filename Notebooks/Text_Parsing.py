#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Scientific Computing
import pandas as pd
import numpy as np

# Text processing.
import re 
import ftfy
import spacy
import nltk
nltk.download('names')
from nltk import ngrams
from nltk.corpus import names
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[8]:


# Initializing tok2vec for lemmatization.
# See "Natural Language Processing in Action" for code reference. 
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler"])

# Customize stop words.
# Some stop words need to be kept, others need to be removed.
words_to_keep = {"not", "never", "more", "less", "very", "well", "together", "keep", "show", "please"}
for word in words_to_keep:
    nlp.Defaults.stop_words.discard(word)
    nlp.vocab[word].is_stop = False

# Add SI noise words that appear in nearly every response.
words_to_add = {"si", "session", "class", "course", "leader", "student", "think", "really"}
for word in words_to_add:
    nlp.Defaults.stop_words.add(word)
    nlp.vocab[word].is_stop = True


# In[2]:


Survey_df = pd.read_csv("../Clean_Data_Resources/Survey_Results.csv")


# In[3]:


# Define all survey columns that are made of text.
Text_Cols = [
    "T_Collaboration_Understanding",
    "T_Leader_Performance_Suggestions",
    "T_Other_Suggestions",
    "T_Program_Overall_Suggestions",
]


# In[4]:


# Tokenize responses.

# Encode responses that functionally do not give information.
Non_Responses = {"n/a", "na", "none", "nothing", "no", "n"}

# Extract nltk's names.
Name_List = set(n.lower() for n in names.words())

def tokenize_response(text):
    # Guard against NaN and empty strings.
    if not isinstance(text, str) or text.strip() == "":
        return []
    # Guard against non-responses.
    if text.strip().lower().replace("/", "") in Non_Responses:
        return []
    # Guard against single character or numeric-only responses.
    if len(text.strip()) <= 1 or text.strip().isdigit():
        return []
    # Fix broken encodings first, then normalize apostrophes and slashes.
    text = ftfy.fix_text(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("/", " ")
    doc = nlp(text.lower())
    # Strip punctuation, whitespace, contractions, emails, and names.
    return [tok.text for tok in doc
            if not tok.is_punct
            and not tok.is_space
            and not tok.is_stop
            and tok.text not in {"n't", "ca", "m", "'m", "ve", "'ve", "re", "'re", "'s", "'d"}
            and not re.match(r'\S+@\S+', tok.text)
            and tok.text not in Name_List]


# In[5]:


# Use an apply function to tokenize all columns in Text_Cols:
for col in Text_Cols:
    Survey_df[col + "_tokens"] = Survey_df[col].apply(tokenize_response)


# In[ ]:


# Function to enact lemmatization.
def lemmatize_response(tokens):
    if not tokens:
        return []
    doc = nlp(" ".join(tokens))
    return [tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space]

# Function to enact ngrams, wher n specifies the n...grams. 
def make_ngrams(tokens, n):
    return [" ".join(gram) for gram in ngrams(tokens, n)]


# In[ ]:


# Lemmatize, then build bigrams and trigrams for each column.
for col in Text_Cols:
    lemma_col = col + "_lemmas"
    Survey_df[lemma_col] = Survey_df[col + "_tokens"].apply(lemmatize_response)

    # Pass n=2 and n=3 to make_ngrams for each lemmatized response.
    Survey_df[col + "_bigrams"]  = Survey_df[lemma_col].apply(lambda x: make_ngrams(x, 2))
    Survey_df[col + "_trigrams"] = Survey_df[lemma_col].apply(lambda x: make_ngrams(x, 3))


# In[ ]:


Survey_df.info()


# # VADER Sentiment

# In[ ]:


# Initialize sentiment analyzer.
Vader_Sentimenter = SentimentIntensityAnalyzer()


# In[ ]:


def vader_score(text, min_tokens=4):
    # Return compound score, skipping responses that are too short to be meaningful.
    if not isinstance(text, str) or text.strip() == "":
        return None
    tokens = text.strip().split()
    if len(tokens) < min_tokens:
        return None
    return Vader_Sentimenter.polarity_scores(text)['compound']

# Run the vader_score funciton with the minimum amount of tokens needed for validity.
for col in Text_Cols:
    Survey_df[col + "_vader"] = Survey_df[col].apply(vader_score)


# In[ ]:


# Initiate an empty list that will hold vader scores for statements that are on a decent length (4 tokens).
Vader_Summary_Non_Null = []

for col in Text_Cols:
    vader_col = col + "_vader"
    # Only include rows where the original text was a non-null, non-empty string.
    has_text = Survey_df[col].notna() & (Survey_df[col].str.strip() != "")
    scores_all = Survey_df.loc[has_text, vader_col]
    # Drop None values returned by token filter.
    scores = scores_all.dropna()
    unscored = has_text.sum() - len(scores)

    Vader_Summary_Non_Null.append({
        # Strip T_ prefix for cleaner display.
        "Column": col.replace("T_", ""),
        "Responses": has_text.sum(),
        "Scored": len(scores),
        "Unscored Too Short": unscored,
        "Mean": round(scores.mean(), 3),
        "Std": round(scores.std(), 3),
        "Neutral": int((scores == 0.0).sum()),
        "Positive": int((scores > 0).sum()),
        "Negative": int((scores < 0).sum()),
        # Responses with strong positive and negative sentiment above VADER's recommended threshold.
        "Strong Positive (>0.5)": int((scores > 0.5).sum()),
        "Strong Negative (<-0.5)": int((scores < -0.5).sum()),
    })

Vader_Summary_Real_df = pd.DataFrame(Vader_Summary_Non_Null).set_index("Column")
print(Vader_Summary_Real_df.to_string())


# In[ ]:


# Sample scored responses across sentiment tiers per column.
print()
for col in Text_Cols:
    vader_col = col + "_vader"
    has_text = Survey_df[col].notna() & (Survey_df[col].str.strip() != "")

    # Restrict to scored responses only.
    scored = Survey_df.loc[has_text].dropna(subset=[vader_col])

    # Define tiers by compound score boundaries.
    # VADER compound scores range from -1.0 to +1.0.
    # Boundaries follow VADER's own recommended thresholds, so  positive: > 0.05 and negative: < -0.05.
    tiers = {
        "Strongly Positive": scored[scored[vader_col] > 0.5].nlargest(1, vader_col),
        "Mildly Positive":   scored[(scored[vader_col] > 0) & (scored[vader_col] <= 0.5)].sample(1, random_state=42),
        "Neutral":           scored[scored[vader_col] == 0.0].sample(1, random_state=42) if (scored[vader_col] == 0.0).sum() > 0 else pd.DataFrame(),
        "Mildly Negative":   scored[(scored[vader_col] < 0) & (scored[vader_col] >= -0.5)].sample(1, random_state=42),
        "Strongly Negative": scored[scored[vader_col] < -0.5].nsmallest(1, vader_col),
    }

    # Print out results.
    print(col)
    for tier, df in tiers.items():
        if len(df) == 0:
            continue
        row = df.iloc[0]
        print(f"{tier} Score = {round(row[vader_col], 3)}")
        print(f"{str(row[col])}")
        print()


# In[ ]:


# Save results as parquet to preserve the bigram, trigram, and lemma columns.
Survey_df["Course_Code"] = Survey_df["Course_Code"].astype(str)
Survey_df.to_parquet("../Clean_Data_Resources/Survey_df_Text_Parsed.parquet")

