import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
df = pd.read_csv("personalized_medication_dataset.csv")

# Step 2: Drop unnecessary columns
df.drop(columns=['Patient_ID', 'BMI'], inplace=True)

# Step 3: Define input features and targets
features = ['Age', 'Gender', 'Weight_kg', 'Height_cm', 'Chronic_Conditions',
            'Drug_Allergies', 'Genetic_Disorders', 'Diagnosis', 'Symptoms']

targets = {
    'medication_model.pkl': ('Recommended_Medication', RandomForestClassifier()),
    'dosage_model.pkl': ('Dosage', RandomForestClassifier()),
    'duration_model.pkl': ('Duration', RandomForestClassifier()),
    'effectiveness_model.pkl': ('Treatment_Effectiveness', RandomForestClassifier()),
    'reactions_model.pkl': ('Adverse_Reactions', RandomForestClassifier()),
    'recovery_model.pkl': ('Recovery_Time_Days', RandomForestRegressor()),
}

# Step 4: Define preprocessing steps
numeric_features = ['Age', 'Weight_kg', 'Height_cm']
categorical_features = ['Gender', 'Chronic_Conditions', 'Drug_Allergies', 'Genetic_Disorders', 'Diagnosis', 'Symptoms']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Make sure model directory exists
os.makedirs('models', exist_ok=True)

# Step 6: Train and save each model
for filename, (target_col, model) in targets.items():
    df_target = df.dropna(subset=[target_col])  # drop rows where target is missing
    X = df_target[features]
    y = df_target[target_col]

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    clf.fit(X, y)
    joblib.dump(clf, f'models/{filename}')
    print(f"✅ Saved model: {filename}")
