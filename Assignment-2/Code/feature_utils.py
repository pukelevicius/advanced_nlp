import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")


def bigram_features(df):
    """
    Add the bigram features to the dataframe
    1. token_bigram: bigram of tokens
    2. POS_bigram: bigram of POS tags
    """
    df['token_bigram'] = df['token'].shift(1, fill_value='_') + ' ' + df['token']
    df['POS_bigram'] = df['POS'].shift(1, fill_value='_') + ' ' + df['POS']
    return df


def trigram_features(df):
    """
    Add the trigram features to the dataframe
    1. token_trigram: trigram of tokens
    2. POS_trigram: trigram of POS tags
    """
    df['token_trigram'] = df['token'].shift(2, fill_value='_') + ' ' + df['token'].shift(1, fill_value='_') + ' ' + df['token']
    df['POS_trigram'] = df['POS'].shift(2, fill_value='_') + ' ' + df['POS'].shift(1, fill_value='_') + ' ' + df['POS']
    return df


# need to fix NER
def ner(df):
    """
    Add the Named Entity Recognition (NER) tags to the dataframe
    """
    ner_tags = []  # Initialize a list to store NER tags
    for index, row in df.iterrows():
        sentence = ' '.join(df[df['sent_id'] == row['sent_id']]['token'])
        doc = nlp(sentence)
        for token in doc:
            ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')  # 'O' for tokens not recognized as named entities
    df['ner'] = ner_tags  # Add the NER tags to the DataFrame
    return df



