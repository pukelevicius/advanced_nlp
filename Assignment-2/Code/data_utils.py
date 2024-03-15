import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, vstack
import re
from feature_utils import *

def read_conll(file_path):
    """
    Read the conll file and return a dataframe
    """
    data = []
    max_cols = 0
    sent_id = ''
    with open(file_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # grab the sentence id (useful for grouping sentences)
            if line.startswith('# sent_id'):
                sent_id = line.split()[-1]
            # skip comments and empty lines
            if line.startswith('#') or line == '\n':
                continue
            line = line.strip().split('\t')
            # add the sentence id to the beginning of the line
            line.insert(0, sent_id)
            data.append(line)
            max_cols = max(max_cols, len(line)) 
        df = pd.DataFrame(data, columns=range(max_cols))
        df.columns = ['sent_id', 'token_id', 'token', 'lemma',
                      'POS', 'Universal_POS', 'morph_type', 'distance_head',
                      'dep_label', 'dep_rel', 'space'] + list(df.columns[11:])
        return df

def conll_transform(df):
    """
    Transform the conll df by duplicating sentences with more than one predicate,
    such that each sentence has exactly one predicate.
    Add a random number to the 'sent_id' column for each duplicated sentence.
    """
    regex = '.*\.0\d'
    multi_predicate_sentence_ids = df.groupby('sent_id').filter(
        lambda x: x.iloc[:, 11].str.match(regex).sum() > 1)['sent_id'].unique()
    rows_to_concat = []
    for sent_id in multi_predicate_sentence_ids:
        sentence = df[df['sent_id'] == sent_id]
        predicate_count = sentence.iloc[:, 11].str.match(regex).sum()
        for i in range(1, predicate_count):
            sentence_copy = sentence.copy(deep=True)
            sentence_copy.iloc[:, 12] = sentence_copy.iloc[:, 12+i]
            # Add a random number to the 'sent_id' column
            sentence_copy['sent_id'] = sentence_copy['sent_id'] + '_' + str(i) # add the duplication number for easier grouping
            rows_to_concat.append(sentence_copy)
    df = pd.concat([df] + rows_to_concat, ignore_index=True)
    df = df.drop(df.columns[13:], axis=1)
    df = df.rename(columns={11: 'predicate', 12: 'argument_type'})
    df = df.fillna('_')
    return df

def prepare_data(file_path):
    """
    Prepare the data by reading the conll file and transforming it and adding features
    """
    df = read_conll(file_path)
    df = conll_transform(df)
    df = bigram_features(df)
    df = trigram_features(df)
    df = get_ner_tags(df)
    df = morph_features(df)
    df = distance_features(df)
    df = extract_verb_type(df)
    df = get_path_from_token_to_predicate(df)

    # feature to indicate if the token is a predicate; maybe redundant
    df['is_token_predicate'] = (df['predicate'] != '_').astype(int)
    # feature for classification task 1: argument identification
    df['is_token_argument'] = (df['argument_type'].str.startswith('ARG')).astype(int)
    # feature for classification task 2: argument classification
    df['argument_label'] = df['argument_type'].apply(lambda x: x if x.startswith('ARG') else 'O')
    return df


def create_count_vectorizer(data, text_feature_columns):
    """
    Create a count vectorizer from the text features in the data
    """
    count_vectorizer = CountVectorizer()
    combined_text = data[text_feature_columns].astype(str).apply(' '.join, axis=1)
    count_vectorizer.fit(combined_text)
    return count_vectorizer

def process_data(data, vectorizer, numeric_features):
    """
    Vectorize the text features and combine with the numeric features
    """
    token_count = vectorizer.transform(data['token'])
    lemma_count = vectorizer.transform(data['lemma'])
    pos_count = vectorizer.transform(data['POS'])
    universal_pos_count = vectorizer.transform(data['Universal_POS'])
    dep_label_count = vectorizer.transform(data['dep_label'])
    dep_rel_count = vectorizer.transform(data['dep_rel'])
    space_count = vectorizer.transform(data['space'])
    predicate_count = vectorizer.transform(data['predicate'])
    ner_count = vectorizer.transform(data['ner'])
    token_bigram_count = vectorizer.transform(data['token_bigram'])
    token_trigram_count = vectorizer.transform(data['token_trigram'])
    pos_bigram_count = vectorizer.transform(data['POS_bigram'])
    pos_trigram_count = vectorizer.transform(data['POS_trigram'])
    path_to_predicate_count = vectorizer.transform(data['path_to_predicate'].astype(str))
    
    X = hstack([token_count, lemma_count, pos_count, universal_pos_count, dep_label_count, dep_rel_count, space_count,
                predicate_count, ner_count, data['is_token_predicate'].values.reshape(-1, 1),
                token_bigram_count, token_trigram_count, pos_bigram_count, pos_trigram_count,
                path_to_predicate_count, data[numeric_features].values])
    
    return X


def calculate_metrics(y_true, y_pred, encoder):
    """
    Calculate precision, recall, and F1 scores separately for is_token_argument and argument_labels.
    Provide aggregated scores using macro averages for each task individually.
    
    Parameters:
    - y_true: The true labels, assumed to be in the same format as y_train/y_test used earlier.
    - y_pred: The predicted labels, in the same format.
    - encoder: The OneHotEncoder instance used for encoding the argument labels.
    
    Returns:
    - Two pandas DataFrames: one for the argument identification task and one for the argument classification task.
    """
    # Argument Identification Metrics
    arg_id_metrics = {
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Score': [
            precision_score(y_true[:, 0], y_pred[:, 0], zero_division=0),
            recall_score(y_true[:, 0], y_pred[:, 0], zero_division=0),
            f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0)
        ]
    }
    arg_id_df = pd.DataFrame(arg_id_metrics)
    
    # Argument Classification Metrics
    arg_class_precision = precision_score(y_true[:, 1:], y_pred[:, 1:], average=None, zero_division=0)
    arg_class_recall = recall_score(y_true[:, 1:], y_pred[:, 1:], average=None, zero_division=0)
    arg_class_f1 = f1_score(y_true[:, 1:], y_pred[:, 1:], average=None, zero_division=0)
    
    arg_class_metrics = pd.DataFrame({
        'Label': encoder.categories_[0],
        'Precision': arg_class_precision,
        'Recall': arg_class_recall,
        'F1 Score': arg_class_f1
    })
    
    # Aggregated (macro) scores for argument classification
    aggregated_class_metrics = pd.DataFrame([{
        'Label': 'Macro Average',
        'Precision': np.mean(arg_class_precision),
        'Recall': np.mean(arg_class_recall),
        'F1 Score': np.mean(arg_class_f1)
    }])
    arg_class_metrics = pd.concat([arg_class_metrics, aggregated_class_metrics], ignore_index=True)
    
    return arg_id_df, arg_class_metrics