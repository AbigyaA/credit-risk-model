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
