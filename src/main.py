"""
main.py
Orchestrates: data generation, training, evaluation, and interpretability.
"""
import os
from data_load import load_data
from model_train import train_and_save
from evaluate import evaluate
from interpret import permutation_importance_plot, pdp_ice_plots

def run_all():
    print('Loading synthetic data and splitting...')
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, random_state=42)

    numeric_features = ['age', 'bmi', 'systolic_bp', 'cholesterol', 'glucose']
    categorical_features = ['activity_level', 'smoker']  # smoker is numeric 0/1, treated as categorical in pipeline

    print('Training model...')
    model_path = os.path.join('models', 'model.joblib')
    pipe = train_and_save(X_train, y_train, numeric_features, categorical_features, model_path=model_path, random_state=42)

    print('Evaluating model on test set...')
    metrics = evaluate(pipe, X_test, y_test)
    print('Evaluation metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')

    print('Computing permutation importance (test set)...')
    permutation_importance_plot(pipe, X_test, y_test, n_repeats=20, random_state=42)

    # PDP + ICE for selected clinically meaningful features
    features = ['age', 'bmi', 'glucose']
    pdp_ice_plots(pipe, X_train, features)

    print('All done. Check the ./figures/ folder for plots and ./models/ for the saved model.')

if __name__ == '__main__':
    run_all()
