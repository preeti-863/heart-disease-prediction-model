# train_and_predict_updated.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('Data_information.csv')  # ensure this file exists in working directory
print("Initial shape:", df.shape)
print(df.head())

# -----------------------------
# Add new clinical columns if missing
# -----------------------------
new_cols_defaults = {
    'ecg': np.nan,                # 0-normal,1-ST-T abnormal,2-LVH
    'cholesterol_value': np.nan,  # mg/dL numeric
    'heart_rate': np.nan,         # bpm
    'chest_pain': np.nan,         # 0-3 categorical
    'diabetes': np.nan,           # 0/1
    'family_history': np.nan,     # 0/1
}

for col, default in new_cols_defaults.items():
    if col not in df.columns:
        df[col] = default
        print(f"Added column '{col}' with default NaN")

# Add BMI column if possible (auto from height & weight)
if 'height' in df.columns and 'weight' in df.columns:
    # height expected in cm, weight in kg
    # avoid zero division
    df['height_m'] = df['height'] / 100.0
    df['bmi'] = df.apply(lambda r: (r['weight'] / (r['height_m']**2)) if (r['height_m'] > 0 and pd.notnull(r['weight'])) else np.nan, axis=1)
    print("Added 'bmi' column computed from height & weight")
else:
    df['bmi'] = np.nan
    print("Height/weight not found; 'bmi' column added as NaN")

# -----------------------------
# Convert / normalize some fields
# -----------------------------
# Some datasets store age in days — convert to years if suspiciously large (> 120)
if 'age' in df.columns:
    # if many ages > 120, assume days
    if (df['age'] > 120).sum() > 0:
        print("Converting 'age' from days to years (age/365).")
        df['age'] = (df['age'] / 365).round().astype(int)

# If categorical cholesterol exists as 1/2/3, create approximate cholesterol_value using mapping
if 'cholesterol' in df.columns and df['cholesterol'].dtype != float and df['cholesterol'].dtype != int:
    # But if cholesterol is strings, try convert to numeric
    try:
        df['cholesterol'] = pd.to_numeric(df['cholesterol'], errors='coerce')
    except:
        pass

if 'cholesterol' in df.columns and df['cholesterol'].isin([1,2,3]).any():
    # mapping: 1 -> ~180, 2 -> ~220, 3 -> ~260 (approx)
    map_cat_to_mg = {1: 180, 2: 220, 3: 260}
    # only fill cholesterol_value where NaN
    df['cholesterol_value'] = df['cholesterol_value'].fillna(df['cholesterol'].map(map_cat_to_mg))
    print("Filled 'cholesterol_value' based on categorical 'cholesterol' (where available)")

# If cholesterol_value column exists with text, convert to numeric
df['cholesterol_value'] = pd.to_numeric(df['cholesterol_value'], errors='coerce')
df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')

# -----------------------------
# Drop unused text columns but keep columns we need
# -----------------------------
drop_cols = []
for c in ['date', 'country', 'occupation', 'id']:
    if c in df.columns:
        drop_cols.append(c)
if drop_cols:
    df = df.drop(columns=drop_cols)
    print("Dropped columns:", drop_cols)

# -----------------------------
# Inspect nulls and basic info
# -----------------------------
print("\nData info after additions:")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# -----------------------------
# Imputation: numerical -> median, categorical-like -> most_frequent
# -----------------------------
# Only include numeric columns that have at least one non-null value (excl. target)
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'disease' and df[col].notnull().any()]


num_imputer = SimpleImputer(strategy='median')
imputed_array = num_imputer.fit_transform(df[numeric_cols])
df_imputed_numeric = pd.DataFrame(imputed_array, columns=numeric_cols, index=df.index)
df[numeric_cols] = df_imputed_numeric  # ✅ fixed assignment

# For any remaining non-numeric columns (rare) -> use most frequent
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[non_numeric] = cat_imputer.fit_transform(df[non_numeric])

print("\nAfter imputation missing values per column:\n", df.isnull().sum())


# -----------------------------
# Prepare features and target
# -----------------------------
# Decide feature set (add the new clinical features)
expected_features = [
    'active', 'age', 'alco', 'ap_hi', 'ap_lo',
    'cholesterol', 'cholesterol_value', 'gluc', 'height', 'weight',
    'gender', 'smoke', 'ecg', 'heart_rate', 'chest_pain',
    'diabetes', 'family_history', 'bmi'
]

# Keep only features present in df; warn if some are missing (but we added most)
features = [f for f in expected_features if f in df.columns]
missing = [f for f in expected_features if f not in df.columns]
if missing:
    print("Warning: Expected features missing from dataset (they will be skipped):", missing)

X = df[features].copy()
y = df['disease'].copy()

print("\nUsing features:", features)
print("X shape:", X.shape, "y shape:", y.shape)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Random Forest with tuned params
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X_train, y_train)

# -----------------------------
# Evaluation helpers
# -----------------------------
def evaluate_model(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))

# Evaluate
y_pred = rf.predict(X_test)
evaluate_model(y_test, y_pred, "Random Forest (final)")

# -----------------------------
# Feature importances
# -----------------------------
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
print("\nTop features:\n", feat_imp_df.head(10))

# Plot top 10 if you want
try:
    top_n = min(10, len(feat_imp_df))
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y='Importance', data=feat_imp_df.head(top_n))
    plt.xticks(rotation=45)
    plt.title('Top feature importances')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plotting skipped (headless env?).", e)

# -----------------------------
# Save model, scaler, and feature list
# -----------------------------
joblib.dump(rf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\nSaved: model.pkl, scaler.pkl, model_features.pkl")

plt.figure(figsize=(6,4))
plt.scatter(range(len(y_test)), y_test, facecolors='none', edgecolors='blue', label='Actual', alpha=0.8)
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted', alpha=0.6)
plt.xlabel('Sample index')
plt.ylabel('Disease (0 = No, 1 = Yes)')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.show()

# -----------------------------
# Interactive user input for prediction
# -----------------------------
def get_user_input():
    try:
        print("\nPlease answer the following patient questions (enter numbers):")
        active = int(input("Are you physically active? (1 = Yes, 0 = No): ") or 0)
        age = int(input("Age (years): ") or 0)
        alco = int(input("Do you consume alcohol? (1 = Yes, 0 = No): ") or 0)
        ap_hi = int(input("Systolic Blood Pressure (ap_hi, mmHg): ") or 120)
        ap_lo = int(input("Diastolic Blood Pressure (ap_lo, mmHg): ") or 80)
        # Offer both options: if user has numeric cholesterol value, prefer that
        cholesterol_value = float(input("Enter total cholesterol (mg/dL), e.g. 180.0: ") or 200.0)
        cholesterol_cat = np.nan  # no longer using category


        gluc = int(input("Glucose (1 = Normal, 2 = Above Normal, 3 = Well Above Normal): ") or 1)
        height = float(input("Height (cm): ") or 160.0)
        weight = float(input("Weight (kg): ") or 60.0)
        gender = int(input("Gender (1 = Female, 2 = Male): ") or 1)
        smoke = int(input("Do you smoke? (1 = Yes, 0 = No): ") or 0)

        # New clinical questions
        ecg = int(input("ECG result (0 = Normal, 1 = ST-T abnormality, 2 = LVH/other): ") or 0)
        heart_rate = int(input("Resting heart rate (bpm), if unknown press Enter: ") or 70)
        chest_pain = int(input("Chest pain type (0 = None, 1 = Atypical, 2 = Non-anginal, 3 = Typical angina): ") or 0)
        diabetes = int(input("Have you been diagnosed with diabetes? (1 = Yes, 0 = No): ") or 0)
        family_history = int(input("Family history of heart disease? (1 = Yes, 0 = No): ") or 0)

        # Compute BMI
        height_m = height / 100.0 if height > 0 else 1
        bmi = weight / (height_m**2) if height_m > 0 else 0.0
        print(f"✅ Calculated BMI: {bmi:.2f}")

        user_row = {
            'active': active,
            'age': age,
            'alco': alco,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            # keep original categorical cholesterol too (if exists)
            'cholesterol': cholesterol_cat if 'cholesterol' in X.columns else np.nan,
            'cholesterol_value': cholesterol_value,
            'gluc': gluc,
            'height': height,
            'weight': weight,
            'gender': gender,
            'smoke': smoke,
            'ecg': ecg,
            'heart_rate': heart_rate,
            'chest_pain': chest_pain,
            'diabetes': diabetes,
            'family_history': family_history,
            'bmi': bmi
        }

        # Keep only columns the model expects, in the same order
        # Load expected features from saved list if available
        try:
            with open('model_features.pkl', 'rb') as f:
                model_features = pickle.load(f)
        except Exception:
            model_features = X.columns.tolist()

        # Build DataFrame and ensure all expected features exist
        user_df = pd.DataFrame([user_row])
        for c in model_features:
            if c not in user_df.columns:
                user_df[c] = 0  # default
        user_df = user_df[model_features]

        return user_df

    except Exception as e:
        print("Error reading input:", e)
        return None

# -----------------------------
# Load saved scaler & model and predict
# -----------------------------
print("\nYou can now provide patient data for a prediction.")
user_data = get_user_input()
if user_data is not None:
    # load scaler and model
    scaler_loaded = joblib.load('scaler.pkl')
    model_loaded = joblib.load('model.pkl')
    # ensure column order matches training features
    with open('model_features.pkl', 'rb') as f:
        model_features = pickle.load(f)

    # scale and predict
    user_data_scaled = scaler_loaded.transform(user_data[model_features])
    pred = model_loaded.predict(user_data_scaled)[0]
    pred_proba = model_loaded.predict_proba(user_data_scaled)[0][1] if hasattr(model_loaded, "predict_proba") else None

    print("\n--- Prediction Result ---")
    if pred == 1:
        print("⚠️ Model predicts: Patient MAY HAVE heart disease.")
    else:
        print("✅ Model predicts: Patient is UNLIKELY to have heart disease.")
    if pred_proba is not None:
        print(f"Model confidence (positive class probability): {pred_proba*100:.2f}%")
else:
    print("No valid input provided. Exiting.")
    
# --- Begin Suggestion Logic ---
suggestions = []

# Extract input values from user_data
input_values = user_data.iloc[0]
age = input_values['age']
bmi = input_values['bmi']
ap_hi = input_values['ap_hi']
ap_lo = input_values['ap_lo']
cholesterol_value = input_values['cholesterol_value']
gluc = input_values['gluc']
smoke = input_values['smoke']
alco = input_values['alco']
active = input_values['active']
heart_rate = input_values.get('heart_rate', 70)
diabetes = input_values.get('diabetes', 0)
family_history = input_values.get('family_history', 0)

# 1. Blood Pressure (Hypertension)
if ap_hi >= 130 or ap_lo >= 80:
    suggestions.append("🩺 *Hypertension detected:* Your blood pressure is above the normal range. This increases your risk of heart disease and stroke. Consider dietary adjustments (e.g., reduce salt), regular physical activity, and consult a healthcare provider.")

# 2. Cholesterol (LDL estimate)
if cholesterol_value >= 240:
    suggestions.append("🧬 *High cholesterol:* Your cholesterol level is in the high-risk category. A low-fat diet, physical activity, and possibly lipid-lowering medications may be advised.")
elif 200 <= cholesterol_value < 240:
    suggestions.append("🧬 *Borderline high cholesterol:* Monitor your levels regularly and consider dietary modifications (less saturated fat, more fiber).")

# 3. Glucose
if gluc == 2:
    suggestions.append("🩸 *Elevated blood glucose:* You may be at risk for prediabetes or metabolic syndrome. Recommend follow-up fasting glucose or HbA1c testing.")
elif gluc == 3:
    suggestions.append("🩸 *High blood glucose:* Strong indicator of possible diabetes. Immediate medical evaluation is advised.")

# 4. BMI
if bmi >= 30:
    suggestions.append("⚖️ *Obesity:* Your BMI falls in the obese range. Obesity is a significant risk factor for cardiovascular disease, hypertension, and diabetes. Consider a structured weight loss plan.")
elif 25 <= bmi < 30:
    suggestions.append("⚖️ *Overweight:* You are above the healthy weight range. Aim to lose 5–10% of body weight through diet and exercise.")

# 5. Smoking
if smoke == 1:
    suggestions.append("🚭 *Smoking risk:* Smoking damages your heart and blood vessels. Quitting reduces your risk of heart disease significantly within 12 months.")

# 6. Alcohol
if alco == 1:
    suggestions.append("🍷 *Alcohol intake:* While moderate drinking is sometimes acceptable, excessive alcohol elevates blood pressure and cholesterol. Keep intake within medical guidelines.")

# 7. Physical Activity
if active == 0:
    suggestions.append("🏃 *Inactive lifestyle:* Regular aerobic exercise (≥150 minutes/week) improves cardiovascular health. Start with walking or cycling 30 minutes daily.")

# 8. Resting Heart Rate
if heart_rate > 100:
    suggestions.append("❤️ *High resting heart rate (tachycardia):* May indicate stress, overtraining, or underlying cardiac issues. If persistent, consult a cardiologist.")
elif heart_rate < 50 and active == 0:
    suggestions.append("❤️ *Low resting heart rate (bradycardia):* This may warrant medical evaluation unless you're an athlete.")

# 9. Diabetes or Family History
if diabetes == 1:
    suggestions.append("🧪 *Known diabetes:* Keep strict control of blood sugar levels to prevent cardiovascular complications.")
if family_history == 1:
    suggestions.append("🧬 *Family history of heart disease:* Your risk is genetically elevated. Regular monitoring of cholesterol, BP, and lifestyle is essential.")

# Final Output
print("\n--- 🩺 Personalized Medical Suggestions ---")
if suggestions:
    for i, tip in enumerate(suggestions, 1):
        print(f"{i}. {tip}")
else:
    print("✅ Your current health inputs do not raise major concerns. Continue maintaining a heart-healthy lifestyle!")
