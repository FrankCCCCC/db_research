# %%
import pandas as pd
import os

# some columns in csv is an array, takes read data distribution for example - [14, 2, 0 ,0]
# on server 0, 14 is what we need
def filter_fn_for_server_0(x):
    return int(x[x.find("[") + 1:x.find(",")])

# read csv from the least header
def read_csv_with_multi_headers(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        
    header_pos = []
    for idx, item in enumerate(lines):
        if 'Transaction ID' in item:
            header_pos.append(idx)
            
    return pd.read_csv(path, skiprows=header_pos[-1])

def get_OU_indice(df_join: pd.DataFrame, ou_id: int):
    ou7_col_names = df_join.columns[['OU7' in col for col in list(df_join.columns)]]
    return ou7_col_names

def get_df_join(path='tx-type/training-data', ou_id: int=None):
    """
    Please follow the code below to get the correct data of server 0
    """

    df_features = pd.read_csv(os.path.join(path, 'transaction-features.csv'))
    df_features = df_features[df_features['Start Time'] > 60000000] # filter out < 60s
    df_features = df_features.set_index('Transaction ID') # for join purpose
    df_features['Tx Type'] = df_features['Tx Type'].astype('int64') # type is an integer which simply adds read write record count
    for col in ['Read Data Distribution', 'Read Data in Cache Distribution', 'Update Data Distribution']:
        df_features[col] = df_features[col].apply(filter_fn_for_server_0)
        df_features[col] = df_features[col].astype('int64')
        
    df_latencies = read_csv_with_multi_headers(os.path.join(path,'transaction-latency-server-0.csv'))
    df_latencies = df_latencies.set_index('Transaction ID').sort_index()
    df_join = df_features.join(df_latencies, how='inner')

    # Interset indice
    interset_indice = pd.Index(df_features.index).intersection(df_latencies.index)
    df_features, df_latencies = df_features.loc[interset_indice], df_latencies.loc[interset_indice]
    df_features, df_latencies = df_features.sort_index(), df_latencies.sort_index()

    if isinstance(ou_id, int):
        ou_col_names = get_OU_indice(df_join=df_join, ou_id=ou_id)
        # df_ou_feats = df_features[ou_col_names]
        df_ou_lats = df_latencies[ou_col_names]
    else:
        df_ou_lats = df_latencies
    df_ou_feats = df_features

    return df_ou_feats, df_ou_lats