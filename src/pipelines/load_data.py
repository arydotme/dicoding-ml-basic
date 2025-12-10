import pandas as pd

def dataRaw():

    dir = "data/01-raw/bank_transactions_data.csv"

    df = pd.read_csv(dir)

    return df

def dataPreprocessing():

    dir = "data/02-preprocessing/data_preprocessing.csv"

    df = pd.read_csv(dir)

    return df

def dataClustering():

    dir = "data/03-clustering/data_clustering.csv"

    df = pd.read_csv(dir)

    return df

def dataInverse():

    dir = "data/04-inverse/data_inverse.csv"

    df = pd.read_csv(dir)

    return df

print("Data berhasil di load")