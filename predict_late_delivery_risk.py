import joblib
import os
import pandas as pd
import sys
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


def load_model(path='late_delivery_risk_model.joblib'):
    if os.path.exists(path):
        return joblib.load(path)

    fallback = 'late_delivery_risk_model_xgb.joblib'
    if os.path.exists(fallback):
        print(f'Warning: default model {path} not found; loading fallback model {fallback}')
        return joblib.load(fallback)

    raise FileNotFoundError(
        f'No model file found. Expected {path} or {fallback}. '
        'Run "py train_late_delivery_model.py" to generate the model locally.'
    )


def predict_batch(model, df):
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    return predictions, probabilities


def evaluate_model(df, model):
    if 'Late_delivery_risk' not in df.columns:
        return None

    X = df[FEATURE_NAMES]
    y = df['Late_delivery_risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred, digits=4),
    }


def format_output(predictions, probabilities, metrics=None):
    lines = ['Late Delivery Risk Predictions', '=' * 50]
    for i, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
        risk_level = 'Late Delivery Risk' if pred == 1 else 'On-Time Delivery'
        prob_late = prob[1] * 100
        lines.append(f'Order {i}: {risk_level} (Late Probability: {prob_late:.1f}%)')

    lines.append(f'\nTotal orders processed: {len(predictions)}')

    if metrics is not None:
        lines.append('')
        lines.append('Evaluation metrics:')
        lines.append(f'Accuracy: {metrics["accuracy"]:.4f}')
        lines.append(f'ROC-AUC: {metrics["roc_auc"]:.4f}')
        lines.append('classification_report:')
        lines.extend(metrics['report'].splitlines())

    return '\n'.join(lines)


def main():
    if len(sys.argv) >= 2 and len(sys.argv) <= 3:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) == 3 else None
        try:
            df = pd.read_csv(csv_path, encoding='cp1252', on_bad_lines='skip', low_memory=False)
            if not all(col in df.columns for col in FEATURE_NAMES):
                error_msg = f'Error: CSV must contain these columns: {FEATURE_NAMES}'
                if output_path:
                    with open(output_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(error_msg + '\n')
                else:
                    print(error_msg)
                return

            model = load_model()
            input_df = df[FEATURE_NAMES]
            predictions, probabilities = predict_batch(model, input_df)
            metrics = evaluate_model(df, model)
            output_text = format_output(predictions, probabilities, metrics)

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(output_text + '\n')
                print(f'Output written to {output_path}')
            else:
                print(output_text)
        except Exception as e:
            error_msg = f'Error reading CSV: {e}'
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(error_msg + '\n')
            else:
                print(error_msg)
    else:
        print('Usage: python predict_late_delivery_risk.py <path_to_csv_file> [output_file]')
        print('CSV must contain columns:', FEATURE_NAMES)


if __name__ == '__main__':
    main()

