import pandas as pd
import re
from feature_utils import *

def read_conll(file_path):
    """
    Read the conll file and return a dataframe
    """
    data = []
    max_cols = 0
    sent_id = ''
    with open(file_path, 'r') as f:
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
    return df

def prepare_data(file_path):
    """
    Prepare the data by reading the conll file and transforming it and adding features
    """
    df = read_conll(file_path)
    df = conll_transform(df)
    return df