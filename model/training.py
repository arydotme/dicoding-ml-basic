#Import Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../data_clustering_inverse.csv")

# Cleaning data in feature target
df.dropna(subset = ['Target'], inplace = True)

# Splitting dataset to train and test
X = df.drop(columns = ['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 0
)

# Get columns with numeric value
num_cols = [cname for cname in X_train.columns if X_train[cname].dtypes in ['int64', 'float64']]

# Get columns with categorical value
cat_cols = [cname for cname in X_train.columns if X_train[cname].dtypes == 'object']

# Combine cols numeric and categorical
my_cols = num_cols + cat_cols
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy()

# Pipeline preprocessing steps
num_transformer = SimpleImputer(strategy = 'mean')

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

my_pipeline = Pipeline(steps=[
    ('transformer', transformer),
    ('model', RandomForestClassifier(n_estimators=100, random_state=0))
])

my_pipeline.fit(X_train, y_train)

# Get predict with model

y_pred = my_pipeline.predict(X_test)