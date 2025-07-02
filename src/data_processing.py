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

import pandas as pd
import numpy as np

def calculate_rfm(df, snapshot_date=None):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',                                          # Frequency
        'Amount': 'sum'                                                    # Monetary
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()

    return rfm

from sklearn.preprocessing import StandardScaler

def normalize_rfm(rfm_df):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    return rfm_scaled

from sklearn.cluster import KMeans

def cluster_rfm(rfm_scaled, random_state=42):
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    clusters = kmeans.fit_predict(rfm_scaled)
    return clusters

def assign_high_risk_label(rfm_df, clusters):
    rfm_df['Cluster'] = clusters

    # Analyze cluster centers
    summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    # Rank by high risk: low frequency and monetary, high recency
    summary['RiskScore'] = summary['Recency'] - summary['Frequency'] - summary['Monetary']
    high_risk_cluster = summary.sort_values('RiskScore', ascending=False).iloc[0]['Cluster']

    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df[['CustomerId', 'is_high_risk']]

def merge_with_main(df, risk_df):
    df = df.merge(risk_df, on='CustomerId', how='left')
    df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)  # Default to 0 if not found
    return df

df['is_high_risk'].value_counts(normalize=True)

import matplotlib.pyplot as plt
import seaborn as sns

rfm_df['Cluster'] = clusters
sns.scatterplot(data=rfm_df, x='Recency', y='Monetary', hue='Cluster')
