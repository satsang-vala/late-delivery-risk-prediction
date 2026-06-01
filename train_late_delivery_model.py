import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding='cp1252', on_bad_lines='skip', low_memory=False)


def build_model_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    try:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', onehot),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    return model


def main():
    data_path = 'APL_Logistics.csv'
    df = load_data(data_path)

    target_column = 'Late_delivery_risk'
    if target_column not in df.columns:
        raise ValueError(f'Target column {target_column} not found in dataset')

    feature_columns = [
        'Shipping Mode',
        'Order Country',
        'Category Name',
        'Order Item Quantity',
        'Days for shipment (scheduled)',
        'Benefit per order',
        'Sales',
    ]

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f'Missing expected feature columns: {missing_features}')

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = [
        'Order Item Quantity',
        'Days for shipment (scheduled)',
        'Benefit per order',
        'Sales',
    ]

    categorical_features = [col for col in feature_columns if col not in numeric_features]

    model = build_model_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print('Model evaluation:')
    print(report)

    joblib.dump(model, 'late_delivery_risk_model.joblib')
    print('Saved trained model to late_delivery_risk_model.joblib')


if __name__ == '__main__':
    main()
