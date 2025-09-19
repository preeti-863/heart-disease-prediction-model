import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Data_information.csv')  # Ensure the CSV file is in the same directory
print("Shape of dataset:", df.shape)
print(df.head())

# Data types and nulls
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Target variable distribution
print("Disease Count")
print(df['disease'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='disease', data=df)
plt.title("Disease Count (0 = No, 1 = Yes)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.boxplot(x='disease', y='age', data=df)
plt.title("Age vs Disease")
plt.show()

# Drop non-numeric and ID columns
df = df.drop(columns=['date', 'country', 'occupation', 'id'])

# Impute missing values (if any)
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split features and target
X = df_imputed.drop('disease', axis=1)
y = df_imputed['disease']

# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, name)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='F1-Score', ascending=False)
print("🔍 Model Comparison:\n")
print(results_df)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

top_n = 10
feat_imp_top = feat_imp_df.head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Importance', data=feat_imp_top, palette='viridis')
plt.title(f'Top {top_n} Important Features (Random Forest)', fontsize=14)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ... all imports and earlier code remain the same up to feature importance plot ...

print("\nTop 5 Important Features:\n", feat_imp_df.head())

# 1️⃣ Train base model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
evaluate_model(y_test, rf.predict(X_test), "Random Forest (Base)")

best_rf = RandomForestClassifier(
    n_estimators=200,   # number of trees
    max_depth=20,       # limit tree depth to avoid overfitting
    random_state=42
)
best_rf.fit(X_train, y_train)

import joblib

# Save the model and scaler
joblib.dump(best_rf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# ✅ Proceed to user input

print("\n✅ Completed training & evaluation. Now entering user input section...")

# --- Input function ---
def get_user_input():
    try:
        active = int(input("Are you physically active? (1 = Yes, 0 = No): "))
        age = int(input("Age (years): "))
        alco = int(input("Do you consume alcohol? (1 = Yes, 0 = No): "))
        ap_hi = int(input("Systolic Blood Pressure (ap_hi): "))
        ap_lo = int(input("Diastolic Blood Pressure (ap_lo): "))
        cholesterol = int(input("Cholesterol (1 = Normal, 2 = Above Normal, 3 = Well Above Normal): "))
        gender = int(input("Gender (1 = Female, 2 = Male): "))
        gluc = int(input("Glucose (1 = Normal, 2 = Above Normal, 3 = Well Above Normal): "))
        height = int(input("Height (in cm): "))
        smoke = int(input("Do you smoke? (1 = Yes, 0 = No): "))
        weight = float(input("Weight (in kg): "))

        return pd.DataFrame([{
            'active': active,
            'age': age,
            'alco': alco,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gender': gender,
            'gluc': gluc,
            'height': height,
            'smoke': smoke,
            'weight': weight
        }])
    except Exception as e:
        print("❌ Error in input:", e)
        return None
    
# Ensure best_rf is already trained earlier in the code
print("\n⚙️ Using the manually tuned Random Forest model (n_estimators=200, max_depth=20).")

# --- Run prediction ---
print("\n📥 Please enter the following health details:")
user_data = get_user_input()

if user_data is not None:
    try:
        # Match the training columns order
        user_data = user_data[X.columns]
        user_data_scaled = scaler.transform(user_data)

        prediction = best_rf.predict(user_data_scaled)[0]
        prediction_proba = best_rf.predict_proba(user_data_scaled)[0][1]

        print("\n🩺 Prediction Result:")
        if prediction == 1:
            print("⚠️ The model predicts that the individual MAY HAVE heart disease.")
        else:
            print("✅ The model predicts that the individual is UNLIKELY to have heart disease.")
        print(f"🧮 Prediction confidence: {prediction_proba*100:.2f}%")

    except Exception as e:
        print(f"❌ Error in prediction: {e}. Please ensure inputs are valid.")
