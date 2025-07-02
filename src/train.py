from sklearn.model_selection import train_test_split

def split_data(df, target='is_high_risk'):
    X = df.drop(columns=[target, 'CustomerId'])  # or any other non-feature cols
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models():
    return {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier()
    }

from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

import mlflow
import mlflow.sklearn

def log_experiment(model_name, model, metrics):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

def main():
    df = pd.read_csv("data/processed/processed_data_with_target.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        print(f"{name} → {metrics}")
        log_experiment(name, model, metrics)

if __name__ == "__main__":
    main()

import joblib

joblib.dump(model, 'models/best_model.pkl')
