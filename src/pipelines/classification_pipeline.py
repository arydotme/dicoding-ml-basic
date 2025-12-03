import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def classify():

    dir = "data/04-inverse/data_inverse.csv"

    df = pd.read_csv(dir)

    X = df.drop(columns = 'Cluster')
    y = df['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state = 42
    )

    cat_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
    num_cols = [cname for cname in X_train.columns if X_train[cname].dtypes in ['int64', 'float64']]

    cols = cat_cols + num_cols
    X_train = X_train[cols].copy()
    X_test = X_test[cols].copy()

    cat_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent', fill_value = 'missing')),
            ('ordinal', OrdinalEncoder())
        ]
    )

    num_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', MinMaxScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', cat_transformer, cat_cols),
            ('num', num_transformer, num_cols)
        ]
    )

    models = {
        'LogisticRegression': LogisticRegression(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    params = {
        'DecisionTreeClassifier': {'max_depth': [8, 16, 32, 64, 128]},
        'RandomForestClassifier': {'n_estimators': [10, 30, 50, 70, 100]}
    }

    results = {}

    for model_name, model in models.items():

        my_pipeline = Pipeline(steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        my_pipeline.fit(X_train, y_train)

        y_pred = my_pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        results[model_name] = {
            'accuracy': acc,
            'model': model
        }

    for name, info in results.items():
        print(f"Model {name} accuracy: {info['accuracy']:.2f}")

if __name__ == '__main__':
    classify()