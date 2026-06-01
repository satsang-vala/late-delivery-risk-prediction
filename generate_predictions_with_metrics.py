import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    'Shipping Mode',
    'Order Country',
    'Category Name',
    'Order Item Quantity',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales',
]

MODEL_PATH = 'late_delivery_risk_model.joblib'
CSV_PATH = 'APL_Logistics.csv'
OUTPUT_PATH = 'predictions.txt'

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH, encoding='cp1252', on_bad_lines='skip', low_memory=False)
    model = joblib.load(MODEL_PATH)

    if not all(col in df.columns for col in FEATURE_NAMES):
        raise ValueError('CSV is missing expected feature columns')

    input_df = df[FEATURE_NAMES]
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)

    lines = ['Late Delivery Risk Predictions', '=' * 50]
    for i, (pred, prob) in enumerate(zip(preds, probs), 1):
        risk_level = 'Late Delivery Risk' if pred == 1 else 'On-Time Delivery'
        lines.append(f'Order {i}: {risk_level} (Late Probability: {prob[1] * 100:.1f}%)')

    lines.append('')
    lines.append(f'Total orders processed: {len(preds)}')
    lines.append('')

    if 'Late_delivery_risk' in df.columns:
        X = df[FEATURE_NAMES]
        y = df['Late_delivery_risk']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        lines.append('Evaluation metrics on holdout test set:')
        lines.append(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        lines.append(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}')
        lines.append('classification_report:')
        lines.extend(classification_report(y_test, y_pred, digits=4).splitlines())

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote combined output to {OUTPUT_PATH}')