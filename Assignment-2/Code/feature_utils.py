import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")


def bigram_features(df):
    """
    Add the bigram features to the dataframe
    1. token_bigram: bigram of tokens
    2. POS_bigram: bigram of POS tags
    """
    df['token_bigram'] = df.groupby('sent_id')['token'].shift(1, fill_value='_') + ' ' + df['token']
    df['POS_bigram'] = df.groupby('sent_id')['POS'].shift(1, fill_value='_') + ' ' + df['POS']
    return df

def trigram_features(df):
    """
    Add the trigram features to the dataframe
    1. token_trigram: trigram of tokens
    2. POS_trigram: trigram of POS tags
    """
    df['token_trigram'] = df.groupby('sent_id')['token'].shift(2, fill_value='_') + ' ' + df.groupby('sent_id')['token'].shift(1, fill_value='_') + ' ' + df['token']
    df['POS_trigram'] = df.groupby('sent_id')['POS'].shift(2, fill_value='_') + ' ' + df.groupby('sent_id')['POS'].shift(1, fill_value='_') + ' ' + df['POS']
    return df

def morph_features(df):
    """
    Extracts morphological features from a DataFrame column containing morphological types.

    Parameters: df : Input DataFrame containing a column named 'morph_type' representing morphological types.

    Returns:
    DataFrame: DataFrame with columns representing extracted morphological features.

    The function extracts the following morphological features and creates new columns in the DataFrame:
    - definite_ind: Indicates whether 'Definite=Ind' is present in the morphological types.
    - number_plur: Indicates whether 'Number=Plur' is present in the morphological types.
    - gender_fem: Indicates whether 'Gender=Fem' is present in the morphological types.
    - case_nom: Indicates whether 'Case=Nom' is present in the morphological types.
    - tense_pres: Indicates whether 'Tense=Pres' is present in the morphological types.
    - mood_ind: Indicates whether 'Mood=Ind' is present in the morphological types.
    - verbform: Represents the verb form based on 'VerbForm' in the morphological types:
        - 1: 'VerbForm=Fin' (Finite verb)
        - 2: 'VerbForm=Inf' (Infinitive)
        - 3: 'VerbForm=Ger' (Gerund)
        - 4: 'VerbForm=Part' (Participle)
        - 0: None of the above
    - voice_passive: Indicates whether 'Voice=Pass' is present in the morphological types.
    - possesive: Indicates whether 'Poss=Yes' is present in the morphological types.
    - pron_type: Represents the pronoun type based on 'PronType' in the morphological types:
        - 1: 'PronType=Prs' (Personal pronoun)
        - 2: 'PronType=Art' (Article)
        - 3: 'PronType=Rel' (Relative pronoun)
        - 4: 'PronType=Int' (Interrogative pronoun)
        - 0: None of the above
    - person: Represents the grammatical person based on 'Person' in the morphological types:
        - 1: 'Person=1' (First person)
        - 2: 'Person=2' (Second person)
        - 3: 'Person=3' (Third person)
        - 0: None of the above

    Example:
    # Create a DataFrame with a 'morph_type' column
    df = pd.DataFrame({'morph_type': ["Definite=Ind|PronType=Art", "PronType=Art|Definite=Def", "Number=Plur"]})
    # Extract morphological features
    df_features = morph_features(df)
    print(df_features)
    """
    df['morph_type'] = df['morph_type'].apply(lambda x: x.split('|'))
    df['definite_ind'] = df['morph_type'].apply(lambda x: 1 if 'Definite=Ind' in x else 0)
    df['number_plur'] = df['morph_type'].apply(lambda x: 1 if 'Number=Plur' in x else 0)
    df['gender_fem'] = df['morph_type'].apply(lambda x: 1 if 'Gender=Fem' in x else 0)
    df['case_nom'] = df['morph_type'].apply(lambda x: 1 if 'Case=Nom' in x else 0)
    df['tense_pres'] = df['morph_type'].apply(lambda x: 1 if 'Tense=Pres' in x else 0)
    df['mood_ind'] = df['morph_type'].apply(lambda x: 1 if 'Mood=Ind' in x else 0)
    df['verbform'] = df['morph_type'].apply(lambda x: 1 if 'VerbForm=Fin' in x else (2 if 'VerbForm=Inf' in x else (3 if 'VerbForm=Ger' in x else (4 if 'VerbForm=Part' in x else 0))))
    df['voice_passive'] = df['morph_type'].apply(lambda x: 1 if 'Voice=Pass' in x else 0)
    df['possesive'] = df['morph_type'].apply(lambda x: 1 if 'Poss=Yes' in x else 0)
    df['pron_type'] = df['morph_type'].apply(lambda x: 1 if 'PronType=Prs' in x else (2 if 'PronType=Art' in x else (3 if 'PronType=Rel' in x else (4 if 'PronType=Int' in x else 0))))
    df['person'] = df['morph_type'].apply(lambda x: 1 if 'Person=1' in x else (2 if 'Person=2' in x else (3 if 'Person=3' in x else 0)))
    return df

def distance_features(df):
    """
    Feature 1: Relative distance to the predicate
        Add the distance from token to predicate feature to the dataframe
        Values:
        1: Close to predicate (within 2 tokens)
        2: Medium distance to predicate (between 3 and 5 tokens)
        3: Far from predicate (more than 5 tokens)
    
    Feature 2: Does the token appear before or after the predicate
        Add the is_before_predicate feature to the dataframe
        Values:
        0: Appears after the predicate
        1: Appears before the predicate
    """
    grouped = df.groupby('sent_id')
    def distance_to_predicate(group):
        predicate_token_id_series = group[group['predicate'] != '_']['token_id']
        if not predicate_token_id_series.empty:
            predicate_token_id = predicate_token_id_series.iloc[0]
            predicate_token_id = int(predicate_token_id) if predicate_token_id != '_' else 0
        else:
            predicate_token_id = 0 # if there is no predicate, set the predicate token id to 0
        group['token_id'] = pd.to_numeric(group['token_id'], errors='coerce')
        group['distance_to_predicate'] = group['token_id'] - predicate_token_id
        group['token_id'] = group['token_id'].astype(int)
        return group
    df = grouped.apply(distance_to_predicate).reset_index(drop=True)
    df['is_before_predicate'] = (df['distance_to_predicate'] < 0).astype(int) # negative distance means the token appears before the predicate
    df['distance_to_predicate'] = df['distance_to_predicate'].abs() # make the distance positive
    df['distance_to_predicate'] = pd.cut(df['distance_to_predicate'], [0, 2, 5, np.inf], labels=[1, 2, 3]).fillna(1)
    return df


def get_ner_tags(df):
    """
    :param df: dataframe to be transformed
    :return: dataframe with NER tags
    """
    def align_and_extract(sentence_df):
        # Reset index for proper alignment
        sentence_df.reset_index(drop=True, inplace=True)
        sentence = ' '.join(sentence_df['token'].tolist())
        doc = nlp(sentence)
        spacy_tokens = [token.text for token in doc]
        data_tokens = sentence_df['token'].tolist()
        spacy_index, data_index = 0, 0
        while data_index < len(data_tokens) and spacy_index < len(spacy_tokens):
            spacy_token = spacy_tokens[spacy_index]
            data_token = data_tokens[data_index]

            if spacy_token == data_token:
                if doc[spacy_index].ent_type_ != '':
                    sentence_df.at[data_index, 'ner'] = doc[spacy_index].ent_type_
                else:
                    sentence_df.at[data_index, 'ner'] = 'O'  # If no entity found, mark as 'O'
                spacy_index += 1
                data_index += 1
            elif spacy_token in data_token:
                spacy_index += 1
            else:
                data_index += 1
        return sentence_df
    grouped = df.groupby('sent_id').apply(align_and_extract)
    grouped['ner'] = grouped['ner'].fillna('O')  # Fill remaining NaNs with 'O'
    return grouped.reset_index(drop=True)
