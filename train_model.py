import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Sample dataset (replace with your actual data)
data = {
    'BMI': [28.5, 32.1, 26.8, 35.0, 29.3, 31.5, 27.9, 33.7],
    'Neck Circumference': [40, 42, 38, 45, 41, 43, 39, 44],
    'Age': [45, 52, 38, 60, 48, 55, 42, 58],
    'Gender': [0, 1, 0, 1, 0, 1, 0, 1],  # 0 for female, 1 for male
    'AHI': [6.2, 18.5, 4.1, 25.3, 7.8, 15.9, 3.5, 22.1],
    'Sleep Apnea': [0, 1, 0, 1, 0, 1, 0, 1]  # 0 for no, 1 for yes
}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[['BMI', 'Neck Circumference', 'Age', 'Gender', 'AHI']]
y = df['Sleep Apnea']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the trained model
model_filename = 'sleep_apnea_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained model saved as {model_filename}")