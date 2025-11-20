import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# ==========================
# 1. Load dataset
# ==========================
df = pd.read_csv("medicine_dataset.csv")
print("Dataset loaded successfully!")

# Make sure these columns exist (adjust names if needed):
# 'Indication'  -> text describing the disease/condition
# 'Name'        -> medicine name
# 'Dosage Form' -> e.g. Tablet, Syrup, Injection
# Optional: 'Strength', 'Category', 'Manufacturer'
df["Indication"] = df["Indication"].astype(str)
df["Name"] = df["Name"].astype(str)

X = df["Indication"]
y = df["Name"]

# ==========================
# 2. Train-test split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 3. Text vectorization
# ==========================
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================
# 4. Train model
# ==========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("\nModel trained successfully!")

# ==========================
# 5. Helper: get medicine info (dosage form, strength, etc.)
# ==========================
def get_medicine_info(med_name):
    """
    Find the first row in df with this medicine name and
    return a dict of important fields (name, dosage form, etc.).
    """
    rows = df[df["Name"] == med_name]

    if rows.empty:
        return {
            "Name": med_name,
            "Dosage Form": "N/A",
            "Strength": "N/A",
            "Category": "N/A",
            "Manufacturer": "N/A",
        }

    row = rows.iloc[0]

    info = {
        "Name": row.get("Name", "N/A"),
        "Dosage Form": row.get("Dosage Form", "N/A") if "Dosage Form" in df.columns else "N/A",
        "Strength": row.get("Strength", "N/A") if "Strength" in df.columns else "N/A",
        "Category": row.get("Category", "N/A") if "Category" in df.columns else "N/A",
        "Manufacturer": row.get("Manufacturer", "N/A") if "Manufacturer" in df.columns else "N/A",
    }
    return info

# ==========================
# 6. Core recommendation logic (top-k medicines with info)
# ==========================
def recommend_medicines_from_symptoms(symptoms, top_k=3):
    """
    Takes symptom text.
    Returns list of dictionaries:
        { 'Name', 'prob', 'Dosage Form', 'Strength', 'Category', 'Manufacturer' }
    """
    vec = vectorizer.transform([symptoms])
    probs = model.predict_proba(vec)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for i in top_indices:
        med_name = model.classes_[i]
        conf = probs[i]

        info = get_medicine_info(med_name)
        info["prob"] = conf
        results.append(info)

    return results

# ==========================
# 7. Format a Q&A style chatbot answer (with dosage form)
# ==========================
def build_chatbot_answer(symptoms, top_k=3):
    """
    Build a user-friendly chatbot message:
    - Echoes symptoms
    - Shows possible medicines with dosage form & other info
    - Adds a strong medical disclaimer
    """
    medicines = recommend_medicines_from_symptoms(symptoms, top_k=top_k)

    if not medicines:
        return (
            "Thanks for sharing your symptoms.\n"
            "Right now I am not able to match any medicines from my database.\n"
            "Please consult a doctor or a nearby clinic for proper diagnosis and treatment."
        )

    lines = []
    lines.append("Thank you for describing your symptoms.")
    lines.append(f"You said: \"{symptoms}\"")
    lines.append("")
    lines.append("Based on similar indications in my training data,")
    lines.append("here are some medicines that are commonly associated with such conditions (information only, NOT a prescription):")

    for med in medicines:
        name = med["Name"]
        prob = med["prob"]
        dosage_form = med["Dosage Form"]
        strength = med["Strength"]
        category = med["Category"]
        manufacturer = med["Manufacturer"]

        lines.append(
            f"- {name} | Dosage form: {dosage_form} | Strength: {strength} | "
            f"Category: {category} | Manufacturer: {manufacturer} "
            f"(confidence: {prob:.2f})"
        )

    lines.append("")
    lines.append("Important:")
    lines.append("- This is NOT a medical diagnosis or a prescription.")
    lines.append("- The correct medicine, dose and duration depend on your age, weight, and medical history.")
    lines.append("- Always consult a registered doctor or qualified healthcare provider before taking any medicine.")
    lines.append("- Do NOT start, stop, or change any medicine based only on this chatbot's suggestions.")

    return "\n".join(lines)

# ==========================
# 8. Chat loop (console demo)
# ==========================
if __name__ == "__main__":
    print("\nHealthcare Medicine Suggestion Chatbot (with Dosage Form) - Demo")
    print("Type your symptoms in one line and I will show possible medicines from my dataset.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You (symptoms): ")
        if user_input.lower().strip() == "exit":
            print("Goodbye! Stay healthy.")
            break

        response = build_chatbot_answer(user_input, top_k=3)
        print("\nBot:\n" + response + "\n" + "=" * 60 + "\n")
