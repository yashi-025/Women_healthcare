import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
merged_data = pd.read_csv("merged_dataset.csv")

numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].mean())

# Encoding categorical variables
le = LabelEncoder()
for col in merged_data.select_dtypes(include=['object']).columns:
    merged_data[col] = le.fit_transform(merged_data[col])

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(merged_data), columns=merged_data.columns)

# Machine Learning Models

## Clustering Model for Patient Grouping
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)

## Classification Model for Disease Prediction
X = df_scaled.drop(columns=['Cluster'])
y = df_scaled['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Model Evaluation
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save Final Processed Dataset
df_scaled.to_csv("final_healthcare_dataset.csv", index=False)

print("Healthcare AI Model training complete. Data processed and saved.")
