from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
import pandas as pd
from src.pipelines.load_data import loadData
from pathlib import Path


def preprocessing():

    df = loadData()

    drop_feature = [
        'TransactionID', 'TransactionDate', 'AccountID', 'DeviceID', 'MerchantID', 'IP Address',
        'PreviousTransactionDate'
    ]

    df = df.drop(drop_feature, axis=1)

    X = df

    num_cols = df.select_dtypes(include = ['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include = 'object').columns

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        verbose_feature_names_out = False
    )

    processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
        processed.toarray() if hasattr(processed, 'toarray') else processed,
        columns = feature_names
    )

    processed_df.to_csv('data/02-preprocessing/processed_data.csv', index=False)

    print("File berhasil disimpan")

    return processed

print("Preprocessing selesai")

if __name__ == "__main__":
    preprocessing()