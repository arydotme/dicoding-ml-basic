#Import Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/04-inverse/data_inverse.csv")

# Splitting dataset to train and test
X = df.drop(columns = ['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_estate = 0
)

# Get columns with numeric value
num_cols = [cname for cname in X_train.columns if X_train[cname].dtypes in ['int64', 'float64']]

# Get columns with categorical value
cat_cols = [cname for cname in X_train.columns if X_train[cname].dtypes == 'object']

# Combine cols numeric and categorical
my_cols = num_cols + cat_cols
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy()

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1))
])

my_pipeline = Pipeline(steps=[
    ('categorical', cat_transformer),
    ('models', RandomForestClassifier(n_estimators=100, random_state=0))
])

my_pipeline.fit(X_train, y_train)

# Get predict with models
y_pred = my_pipeline.predict(X_test)

# Model evaluation
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, cmap='Blues', annot=True)
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt. g('../src/imgs/confusion_matrix.png')

print(classification_report(y_test, y_pred))