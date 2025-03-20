# Thyroid Cancer Outlier Detection using LOF

## 📌 Project Overview
This project detects **outliers (potential thyroid cancer cases)** using **Local Outlier Factor (LOF)**, a density-based anomaly detection method. The dataset contains medical attributes such as **TSH, T3, TT4, T4U, and FTI** that help identify unusual patterns in thyroid health.

## 🚀 Features
- **Loads and preprocesses thyroid patient data**
- **Applies Local Outlier Factor (LOF)** for outlier detection
- **Displays detected outliers in table format (no graphical output)**
- **Saves results in `thyroid_outliers_detected.csv`** for further analysis

## 📂 Dataset
The dataset (`ann-test.csv`) contains:
- **Numerical Features**: `Age`, `TSH`, `T3_measured`, `TT4_measured`, `T4U_measured`, `FTI_measured`
- **Categorical/Binary Features**: Various medical conditions (0/1 values)
- **Outlier Label**: A column indicating known outliers (for validation)

## 📦 Installation
Before running the script, install the required dependencies:

```bash
pip install pandas numpy scikit-learn
```

## ⚙️ How It Works
1. **Load the dataset** from `ann-test.csv`
2. **Extract numerical features** for analysis
3. **Standardize the data** using `StandardScaler`
4. **Apply LOF algorithm** to detect outliers
5. **Filter outliers** (LOF score = -1) and display them in table format
6. **Save the detected outliers** into `thyroid_outliers_detected.csv`

## 📝 Code Usage
Run the script:
```bash
python thyroid_outlier_lof.py
```

### **Expected Output (Table View)**
| Age  | TSH   | T3_measured | TT4_measured | T4U_measured | FTI_measured | LOF_Score | LOF_Anomaly_Score |
|------|-------|------------|-------------|-------------|-------------|-----------|----------------|
| 65   | 30.21 | 0.15       | 220         | 0.89        | 42          | -1        | -2.13          |
| 45   | 0.02  | 1.45       | 99          | 0.50        | 12          | -1        | -1.98          |
| 78   | 10.50 | 0.25       | 150         | 0.75        | 30          | -1        | -2.24          |

## 📊 Results
- **LOF Score = -1** → Marked as **outlier**
- **Lower LOF Anomaly Score** → More extreme anomaly
- **Results saved in** `thyroid_outliers_detected.csv`

## 🛠️ Next Steps
- **Parameter tuning**: Adjust `n_neighbors` and `contamination` for improved accuracy
- **Feature Engineering**: Include additional medical parameters
- **Validation**: Compare results with medically labeled outliers

## 👨‍💻 Author
Developed by **Vikram** for **Thyroid Cancer Outlier Detection** project.

---

This repository contains a **submission-ready** project! 🚀 If you need modifications, feel free to reach out. 😊

