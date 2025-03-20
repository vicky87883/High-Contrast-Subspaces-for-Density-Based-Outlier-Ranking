import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Compare LOF outliers with original dataset labels
comparison_df = df[['Outlier_label', 'LOF_Score', 'LOF_Anomaly_Score']].copy()
lof_vs_labels = comparison_df.groupby(['Outlier_label', 'LOF_Score']).size().unstack()

# Heatmap of LOF Outliers vs Dataset Labels
plt.figure(figsize=(8, 5))
sns.heatmap(lof_vs_labels, annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.xlabel("LOF Prediction (-1: Outlier, 1: Normal)")
plt.ylabel("Original Outlier Label")
plt.title("Comparison of LOF Outliers vs Dataset Labels")
plt.show()

# Scatter plot to visualize outliers in two key features
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['TSH'], y=df['FTI_measured'], hue=df['LOF_Score'], palette={1: 'blue', -1: 'red'})
plt.xlabel("TSH Levels")
plt.ylabel("FTI Levels")
plt.title("Outlier Detection using LOF (Red = Outlier)")
plt.legend(title="LOF Score")
plt.show()

# Save the output as a new CSV file
df.to_csv("thyroid_outliers_detected.csv", index=False)
print("Outlier detection complete! Results saved in 'thyroid_outliers_detected.csv'.")
