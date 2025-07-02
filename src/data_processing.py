import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def create_aggregated_features(df):
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Value': ['sum', 'mean'],
        'TransactionStartTime': ['min', 'max']
    })

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)

    return agg_df

def extract_time_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year

    return df

categorical_cols = ['ChannelId', 'ProductCategory', 'PricingStrategy']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_cols = ['Amount', 'Value']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

from sklearn.pipeline import Pipeline

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

('scaler', StandardScaler())

from sklearn.preprocessing import MinMaxScaler
