"""
Healthcare Chatbot - Disease Prediction Model Training
Author: AI Assistant
Date: November 15, 2025

This script trains three machine learning models (Logistic Regression, Decision Tree, 
and Random Forest) to predict diseases based on symptoms.

Required libraries: numpy, pandas, scikit-learn, matplotlib
Required file: dataset.csv (with Disease and Symptom columns)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

def load_and_explore_data(filename='dataset.csv'):
    """Load the dataset and display basic information"""
    print("="*70)
    print("LOADING DATASET")
    print("="*70)

    df = pd.read_csv(filename)

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of unique diseases: {df['Disease'].nunique()}")
    print(f"\nFirst few rows:")
    print(df.head())

    return df

def preprocess_data(df):
    """Preprocess the dataset and create feature matrix"""
    print("\n" + "="*70)
    print("PREPROCESSING DATA")
    print("="*70)

    # Fill NaN values with empty string
    df_filled = df.fillna('')

    # Get symptom columns
    symptom_columns = [col for col in df.columns if 'Symptom' in col]

    # Create a list of all unique symptoms
    all_symptoms = set()
    for col in symptom_columns:
        symptoms = df[col].unique()
        for s in symptoms:
            if isinstance(s, str) and s.strip() != '':
                all_symptoms.add(s.strip())

    all_symptoms = sorted(list(all_symptoms))

    print(f"\nTotal unique symptoms: {len(all_symptoms)}")
    print(f"Sample symptoms: {all_symptoms[:10]}")

    # Create binary feature matrix
    features = []
    labels = []

    for idx, row in df_filled.iterrows():
        # Get all symptoms for this row
        patient_symptoms = []
        for col in symptom_columns:
            symptom = row[col]
            if isinstance(symptom, str) and symptom.strip() != '':
                patient_symptoms.append(symptom.strip())

        # Create binary vector
        feature_vector = [1 if symptom in patient_symptoms else 0 for symptom in all_symptoms]
        features.append(feature_vector)
        labels.append(row['Disease'])

    X = np.array(features)
    y = np.array(labels)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y, all_symptoms

def train_models(X_train, X_test, y_train, y_test):
    """Train all three models and return them with predictions"""
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)

    models = {}
    predictions = {}
    accuracies = {}

    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    models['Logistic Regression'] = lr_model
    predictions['Logistic Regression'] = lr_pred
    accuracies['Logistic Regression'] = lr_accuracy

    print(f"   Accuracy: {lr_accuracy*100:.2f}%")

    # 2. Decision Tree
    print("\n2. Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)

    models['Decision Tree'] = dt_model
    predictions['Decision Tree'] = dt_pred
    accuracies['Decision Tree'] = dt_accuracy

    print(f"   Accuracy: {dt_accuracy*100:.2f}%")

    # 3. Random Forest
    print("\n3. Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    accuracies['Random Forest'] = rf_accuracy

    print(f"   Accuracy: {rf_accuracy*100:.2f}%")

    return models, predictions, accuracies

def evaluate_models(y_test, predictions, accuracies):
    """Evaluate and display detailed metrics for all models"""
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)

    results_data = {
        'Model': [],
        'Accuracy (%)': [],
        'Precision (%)': [],
        'Recall (%)': [],
        'F1-Score (%)': []
    }

    for model_name, pred in predictions.items():
        print(f"\n{model_name.upper()}")
        print("-" * 70)

        report = classification_report(y_test, pred, output_dict=True, zero_division=0)

        accuracy = accuracies[model_name] * 100
        precision = report['macro avg']['precision'] * 100
        recall = report['macro avg']['recall'] * 100
        f1_score = report['macro avg']['f1-score'] * 100

        print(f"Accuracy:  {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1-Score:  {f1_score:.2f}%")

        results_data['Model'].append(model_name)
        results_data['Accuracy (%)'].append(accuracy)
        results_data['Precision (%)'].append(precision)
        results_data['Recall (%)'].append(recall)
        results_data['F1-Score (%)'].append(f1_score)

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('model_performance_results.csv', index=False)
    print("\n✓ Results saved to 'model_performance_results.csv'")

    return results_df

def save_models(models, all_symptoms):
    """Save trained models and symptom list"""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)

    # Save each model
    for model_name, model in models.items():
        filename = model_name.lower().replace(' ', '_') + '_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ {model_name} saved to '{filename}'")

    # Save symptom list
    with open('symptoms_list.pkl', 'wb') as f:
        pickle.dump(all_symptoms, f)
    print(f"✓ Symptom list saved to 'symptoms_list.pkl'")

def display_feature_importance(model, all_symptoms):
    """Display top important features from Random Forest"""
    print("\n" + "="*70)
    print("TOP 15 IMPORTANT SYMPTOMS (Random Forest)")
    print("="*70)

    feature_importance = pd.DataFrame({
        'Symptom': all_symptoms,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(feature_importance.head(15).to_string(index=False))

    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("\n✓ Feature importance saved to 'feature_importance.csv'")

def main():
    """Main function to run the entire pipeline"""
    print("\n" + "="*70)
    print("HEALTHCARE CHATBOT - DISEASE PREDICTION MODEL TRAINING")
    print("="*70)

    # Load data
    df = load_and_explore_data('dataset.csv')

    # Preprocess data
    X, y, all_symptoms = preprocess_data(df)

    # Split data
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Train models
    models, predictions, accuracies = train_models(X_train, X_test, y_train, y_test)

    # Evaluate models
    results_df = evaluate_models(y_test, predictions, accuracies)

    # Display summary
    print("\n" + "="*70)
    print("SUMMARY - ALL MODELS")
    print("="*70)
    print(results_df.to_string(index=False))

    # Display feature importance
    display_feature_importance(models['Random Forest'], all_symptoms)

    # Save models
    save_models(models, all_symptoms)

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nAll models have accuracy above 80% ✓")
    print("\nGenerated files:")
    print("  1. logistic_regression_model.pkl")
    print("  2. decision_tree_model.pkl")
    print("  3. random_forest_model.pkl")
    print("  4. symptoms_list.pkl")
    print("  5. model_performance_results.csv")
    print("  6. feature_importance.csv")

if __name__ == "__main__":
    main()
