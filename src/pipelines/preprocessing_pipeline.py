from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pandas as pd
from src.pipelines.load_data import dataRaw
import joblib

def preprocessing():

    df = dataRaw()

    drop_feature = [
        'TransactionID', 'TransactionDate', 'AccountID', 'DeviceID', 'MerchantID', 'IP Address',
        'PreviousTransactionDate'
    ]

    df = df.drop(drop_feature, axis=1)

    X = df

    num_cols = df.select_dtypes(include = ['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include = ['object']).columns

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('ordinal', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)),
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        verbose_feature_names_out = False
    )

    my_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
    ])

    processed = my_pipeline.fit_transform(X)

    feature_names = my_pipeline.get_feature_names_out()

    processed_df = pd.DataFrame(
        processed.toarray() if hasattr(processed, 'toarray') else processed,
        columns = feature_names
    )

    processed_df.to_csv('data/02-preprocessing/data_preprocessing.csv', index=False)

    joblib.dump(my_pipeline, 'data/02-preprocessing/pipeline.pkl')

    print("File berhasil disimpan")

    return processed

print("Preprocessing selesai")

if __name__ == "__main__":
    preprocessing()