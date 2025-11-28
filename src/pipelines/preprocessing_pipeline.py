from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
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

    try:
        inverse = my_pipeline.named_steps['preprocessor']

        num_inverse = inverse.named_transformers_['num']
        cat_inverse = inverse.named_transformers_['cat']

        num_proces = processed[:, :len(num_cols)]
        cat_proces = processed[:, :len(cat_cols)]

        num_inverse_scaled = num_inverse.named_steps['scaler'].inverse_transform(num_proces)

        df_inverse_num = pd.DataFrame(
            num_inverse_scaled,
            columns = num_cols
        )

        cat_inverse_encode = cat_inverse.named_steps['ordinal'].inverse_transform(cat_proces)

        df_inverse_cat = pd.DataFrame(
            cat_inverse_encode,
            columns = cat_cols
        )

        df_inverse = pd.concat([df_inverse_num, df_inverse_cat], axis=1)

        df_inverse = df_inverse[df.columns]

        df_inverse.to_csv('data/04-inverse/data_inverse.csv', index=False)

    except Exception as e:
        print(f"error: {e}")

    print("File berhasil disimpan")

    return processed

print("Preprocessing selesai")

if __name__ == "__main__":
    preprocessing()