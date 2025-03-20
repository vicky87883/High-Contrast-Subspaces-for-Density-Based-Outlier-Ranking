import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# Load dataset
df = pd.read_csv("ann-test.csv")  # Ensure file is in the same directory

# Select relevant features for outlier detection
features = ['Age', 'TSH', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']
X = df[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
df['LOF_Score'] = lof.fit_predict(X_scaled)  # -1 for outliers, 1 for normal points
df['LOF_Anomaly_Score'] = lof.negative_outlier_factor_  # Lower = more outlier-like

# Filter outliers and display them
df_outliers = df[df['LOF_Score'] == -1]
print("Outliers Detected:")
print(df_outliers)

# Save the output as a new CSV file
df_outliers.to_csv("thyroid_outliers_detected.csv", index=False)
print("Outlier detection complete! Results saved in 'thyroid_outliers_detected.csv'.")
