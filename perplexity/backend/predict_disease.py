"""
Healthcare Chatbot - Disease Prediction with Details
Author: AI Assistant
Date: November 15, 2025

This script loads the trained models and predicts diseases based on input symptoms.
It also shows disease descriptions, precautions, and symptom details.

Required files: 
- One of the model files (.pkl)
- symptoms_list.pkl
- symptom_Description.csv
- symptom_precaution.csv
- Symptom-severity.csv
"""

import pickle
import numpy as np
import pandas as pd

def load_data_and_models():
    """Load models, symptom list, and additional data files"""
    # Load model and symptom list
    model, symptoms_list = load_model_and_symptoms()
    
    # Load description, precaution, and severity data
    symptom_desc_df = pd.read_csv('backend\symptom_Description.csv')
    symptom_desc_dict = dict(zip(symptom_desc_df['Disease'], symptom_desc_df['Description']))
    
    symptom_precaution_df = pd.read_csv('backend\symptom_precaution.csv')
    precaution_dict = {}
    for idx, row in symptom_precaution_df.iterrows():
        disease = row['Disease']
        precautions = [row[f'Precaution_{i}'] for i in range(1, 5) if pd.notna(row[f'Precaution_{i}'])]
        precaution_dict[disease] = precautions
    
    symptom_severity_df = pd.read_csv('backend\Symptom-severity.csv')
    severity_dict = dict(zip(symptom_severity_df['Symptom'], symptom_severity_df['weight']))
    
    return model, symptoms_list, symptom_desc_dict, precaution_dict, severity_dict

def load_model_and_symptoms(model_name='random_forest'):
    """Load the trained model and symptom list"""
    model_file = f'{model_name}_model.pkl'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open('symptoms_list.pkl', 'rb') as f:
        symptoms_list = pickle.load(f)
    
    print(f"✓ Model '{model_name}' loaded successfully!")
    print(f"✓ Symptom list loaded ({len(symptoms_list)} symptoms)")
    
    return model, symptoms_list

def create_symptom_vector(input_symptoms, all_symptoms):
    """Convert input symptoms to binary feature vector"""
    input_symptoms = [s.strip().lower() for s in input_symptoms]
    feature_vector = []
    for symptom in all_symptoms:
        if symptom.lower() in input_symptoms:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    return np.array(feature_vector).reshape(1, -1)

def predict_disease_with_details(model, symptoms_list, symptom_desc_dict, precaution_dict, severity_dict, input_symptoms):
    """Predict disease and return detailed information"""
    # Create feature vector
    X = create_symptom_vector(input_symptoms, symptoms_list)
    
    # Predict
    prediction = model.predict(X)[0]
    
    # Get probability if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        max_prob = max(probabilities) * 100
        confidence = max_prob
    
    # Get disease details
    disease_desc = symptom_desc_dict.get(prediction, "No description available")
    disease_precautions = precaution_dict.get(prediction, ["No precautions available"])
    
    # Get symptom severity details
    symptom_severity = []
    for symptom in input_symptoms:
        severity = severity_dict.get(symptom, "No severity data")
        symptom_severity.append(f"{symptom}: {severity}")
    
    return prediction, confidence, disease_desc, disease_precautions, symptom_severity

def main():
    """Main function for disease prediction"""
    print("="*70)
    print("HEALTHCARE CHATBOT - DISEASE PREDICTION WITH DETAILS")
    print("="*70)
    
    # Load all data
    model, symptoms_list, symptom_desc_dict, precaution_dict, severity_dict = load_data_and_models()
    
    print("\n" + "="*70)
    print("AVAILABLE SYMPTOMS")
    print("="*70)
    print("Sample symptoms (first 10):")
    for i, symptom in enumerate(symptoms_list[:10], 1):
        print(f"{i}. {symptom}")
    print(f"... and {len(symptoms_list) - 10} more symptoms")
    
    print("\n" + "="*70)
    print("ENTER SYMPTOMS")
    print("="*70)
    print("Enter symptoms separated by commas (e.g., itching, skin_rash, fever)")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Symptoms: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThank you for using Healthcare Chatbot!")
            break
        
        if not user_input:
            print("Please enter at least one symptom.\n")
            continue
        
        # Parse symptoms
        input_symptoms = [s.strip() for s in user_input.split(',')]
        
        print(f"\nYou entered: {', '.join(input_symptoms)}")
        
        # Predict
        prediction, confidence, disease_desc, disease_precautions, symptom_severities = predict_disease_with_details(
            model, symptoms_list, symptom_desc_dict, precaution_dict, severity_dict, input_symptoms
        )
        
        print("\n" + "="*70)
        print("PREDICTION RESULT")
        print("="*70)
        print(f"\nPredicted Disease: {prediction}")
        if confidence:
            print(f"Confidence: {confidence:.2f}%")
        
        print(f"\nDisease Description:")
        print(f"  {disease_desc}")
        
        print(f"\nRecommended Precautions:")
        for i, precaution in enumerate(disease_precautions, 1):
            print(f"  {i}. {precaution}")
        
        print(f"\nSymptom Severity Details:")
        for severity in symptom_severities:
            print(f"  {severity}")
        
        print("\n" + "-"*70)
        print("Note: This is a prediction based on machine learning.")
        print("Please consult a healthcare professional for proper diagnosis.")
        print("-"*70 + "\n")

if __name__ == "__main__":
    main()
