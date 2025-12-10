from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from src.pipelines.load_data import loadData

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
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    processed = preprocessor.fit_transform(X)

    processed_df = pd.DataFrame(
        processed.toarray() if hasattr(processed, 'toarray') else processed
    )

    processed_df.to_csv('./data/preprocessed_data.csv', index=False)

    return processed

if __name__ == "__main__":
    preprocessing()
