import pandas as pd

def loadData():

    dir = "data/01-raw/bank_transactions_data.csv"

    df = pd.read_csv(dir)

    return df

print("Data berhasil di load")