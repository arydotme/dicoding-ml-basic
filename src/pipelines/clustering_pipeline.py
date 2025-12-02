from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import matplotlib.pyplot as plt
import joblib

dir = "data/02-preprocessing/data_preprocessing.csv"

def nCluster():

    df = pd.read_csv(dir)

    km = KMeans(random_state = 42)
    n = KElbowVisualizer(km, k = (1, 10))

    n.fit(df)
    n.finalize()

    plt.savefig("src/imgs/KElbowVisualizer.png")

def clustering():

    df = pd.read_csv(dir)

    model = KMeans(n_clusters=3, random_state=42, n_init="auto")

    df['Cluster'] = model.fit_predict(df)

    df.to_csv("data/03-clustering/data_clustering.csv", index=False)

    joblib.dump(model, "data/03-clustering/clustering.pkl")

if __name__ == '__main__':
    nCluster()
    clustering()