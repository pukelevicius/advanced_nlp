import pandas as pd

def read_conllu(file_path):
    data = []
    max_cols = 0
    sent_id = ''
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('# newdoc'):
                sent_id = line.split()[-1]
            if line.startswith('#') or line == '\n':
                continue
            line = line.strip().split('\t')
            line.insert(0, sent_id)
            data.append(line)
            max_cols = max(max_cols, len(line)) 
        df = pd.DataFrame(data, columns=range(max_cols))
        df.columns = ['sent_id', 'token_id', 'token', 'lemma',
                      'POS', 'Universal_POS', 'morph_type', 'distance_head',
                      'dep_label', 'dep_rel', 'space'] + list(df.columns[11:]) # need to fix the remaining column names
        return df