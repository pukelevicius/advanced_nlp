import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
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
    """
    # regex to match predicate
    regex = '.*\.0\d'
    # Group by sentence id and filter sentences with more than one predicate
    multi_predicate_sentence_ids = df.groupby('sent_id').filter(
        lambda x: x.iloc[:, 11].str.match(regex).sum() > 1)['sent_id'].unique()
    # Accumulate duplicate rows to be concatenated in a list
    rows_to_concat = []
    for sent_id in multi_predicate_sentence_ids:
        sentence = df[df['sent_id'] == sent_id]
        # get count of predicates
        predicate_count = sentence.iloc[:, 11].str.match(regex).sum()
        # duplicate the sentence for each predicate and copy the 13th column to the 12th column and so on
        for i in range(1, predicate_count):
            sentence_copy = sentence.copy(deep=True)
            sentence_copy.iloc[:, 12] = sentence_copy.iloc[:, 12+i]
            rows_to_concat.append(sentence_copy)
    # Concatenate the original and duplicated sentences
    df = pd.concat([df] + rows_to_concat, ignore_index=True)
    # drop columns starting from 13th
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
    
    X = hstack([token_count, lemma_count, pos_count, universal_pos_count, dep_label_count, dep_rel_count, space_count,
                predicate_count, ner_count, data['is_token_predicate'].values.reshape(-1, 1),
                token_bigram_count, token_trigram_count, pos_bigram_count, pos_trigram_count, data[numeric_features].values])
    
    return X


def calculate_metrics(start_label, end_label, y_dev_encoded_array, y_pred):
    """
    Calculate metrics for the classification task
    For labels 0 and 1: is_token_argument (argument identification)
    For labels 2 to 30: argument_label (argument classification)
    """
    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0
    total_support = 0
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(start_label, end_label):
        precision, recall, f1, support = precision_recall_fscore_support(y_dev_encoded_array[:, i], y_pred[:, i], average=None, zero_division=0)
        for j in range(len(precision)):
            weighted_precision_sum += precision[j] * support[j]
            weighted_recall_sum += recall[j] * support[j]
            weighted_f1_sum += f1[j] * support[j]
            precisions.append(precision[j])
            recalls.append(recall[j])
            f1_scores.append(f1[j])
        total_support += sum(support)

    weighted_avg_precision = weighted_precision_sum / total_support if total_support > 0 else 0
    weighted_avg_recall = weighted_recall_sum / total_support if total_support > 0 else 0
    weighted_avg_f1 = weighted_f1_sum / total_support if total_support > 0 else 0

    macro_avg_precision = np.mean(precisions)
    macro_avg_recall = np.mean(recalls)
    macro_avg_f1 = np.mean(f1_scores)
    return weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, macro_avg_precision, macro_avg_recall, macro_avg_f1
