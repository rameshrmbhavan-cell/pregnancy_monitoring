import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD DATA
# The 'comment' parameter skips the metadata at the top of your pregnancy.csv
df = pd.read_csv('pregnancy.csv', comment='#', skipinitialspace=True)

# 2. DATA CLEANING
# Keep only the rows that have a 'fetal_health' score
df = df.dropna(subset=['fetal_health']).copy()

# Fix columns that might have mixed text and numbers
cols_to_fix = ['light_decelerations', 'prolongued_decelerations']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows that are still empty after cleaning
df = df.dropna()

# 3. SPLIT DATA INTO TRAIN AND TEST
X = df.drop('fetal_health', axis=1) # Features (the health data)
y = df['fetal_health']              # Target (the result we want to predict)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TRAIN THE AI MODEL
print("Training the AI model... Please wait.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. EVALUATE PERFORMANCE
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\n--- Detailed Performance Report ---")
print(classification_report(y_test, y_pred))

# 6. VISUALIZATION 1: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('AI Accuracy: Actual vs Predicted')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('accuracy_chart.png') # Saves to your folder
print("\nChart saved: accuracy_chart.png")

# 7. VISUALIZATION 2: Feature Importance
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importances.tail(10).plot(kind='barh', color='teal')
plt.title('Top 10 Factors Influencing Fetal Health')
plt.tight_layout()
plt.savefig('key_factors.png') # Saves to your folder
print("Chart saved: key_factors.png")

print("\nAI Run Complete!")
# To predict a single case (example data)
import numpy as np
sample_data = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample_data)
print(f"The AI predicts this baby's health status is: {prediction[0]}")
# To predict a single case WITHOUT the warning
# We convert the single row back into a DataFrame with names
sample_row = X_test.iloc[[0]] 
prediction = model.predict(sample_row)

status_map = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}
print(f"The AI predicts this baby's health status is: {status_map[prediction[0]]}")