import joblib
import pandas as pd
import numpy as np


def inverse():
    try:
        # Load data yang sudah diproses
        df_processed = pd.read_csv("data/02-preprocessing/data_preprocessing.csv")
        my_pipeline = joblib.load("data/02-preprocessing/pipeline.pkl")
        cluster_labels = pd.read_csv("data/03-clustering/data_clustering.csv")['Cluster'].values

        # Dapatkan preprocessor
        preprocessor = my_pipeline.named_steps['preprocessor']

        # Dapatkan kolom numerik dan kategorikal dari pipeline
        num_cols = preprocessor.transformers_[0][2]
        cat_cols = preprocessor.transformers_[1][2]

        #Dapatkan transformer
        num_transformer = preprocessor.named_transformers_['num']
        cat_transformer = preprocessor.named_transformers_['cat']

        # Ambil data numerik dan kategorikal dari df_processed
        num_data = df_processed[num_cols].values if all(col in df_processed.columns for col in num_cols) else \
        df_processed.iloc[:, :len(num_cols)].values
        cat_data = df_processed[cat_cols].values if all(col in df_processed.columns for col in cat_cols) else \
        df_processed.iloc[:, len(num_cols):].values

        # Inverse transform numerik
        num_inverse_scaled = num_transformer.named_steps['scaler'].inverse_transform(num_data)

        num_round = np.round(num_inverse_scaled, 3)

        df_inverse_num = pd.DataFrame(
            num_round,
            columns=num_cols
        )

        # Inverse transform kategorikal
        cat_inverse_encode = cat_transformer.named_steps['ordinal'].inverse_transform(cat_data)

        df_inverse_cat = pd.DataFrame(
            cat_inverse_encode,
            columns=cat_cols
        )

        df_inverse = pd.concat([df_inverse_num, df_inverse_cat], axis=1)

        df_inverse['Cluster'] = cluster_labels

        df_inverse.to_csv('data/04-inverse/data_inverse.csv', index=False)

        print("File berhasil di simpan")

        return df_inverse

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    inverse()