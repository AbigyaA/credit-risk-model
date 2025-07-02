from src.data_processing import create_aggregated_features


def test_create_aggregated_features():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 50],
        'Value': [100, 200, 50],
        'TransactionStartTime': pd.to_datetime(['2021-01-01', '2021-01-05', '2021-02-01'])
    })
    result = create_aggregated_features(df)
    assert 'Amount_sum' in result.columns
    assert result.shape[0] == 2

def test_split_data():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [1, 0, 1, 0],
        'is_high_risk': [0, 1, 0, 1]
    })
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 3
    assert len(y_test) == 1

def test_model_training():
    from sklearn.linear_model import LogisticRegression
    X = [[0, 1], [1, 0], [0, 0], [1, 1]]
    y = [0, 1, 0, 1]
    model = LogisticRegression().fit(X, y)
    assert model.score(X, y) > 0.5
