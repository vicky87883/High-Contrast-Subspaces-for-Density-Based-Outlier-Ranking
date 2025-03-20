from sklearn.ensemble import IsolationForest
from outlier_analysis.views import fetch_data_from_db, preprocess_data
import psycopg2

def detect_outliers():
    """Detect outliers and update database"""
    df = fetch_data_from_db()
    df_scaled = preprocess_data(df)

    # Apply Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df_scaled["outlier"] = iso_forest.fit_predict(df_scaled)

    # Convert labels (-1 to 1)
    df_scaled["outlier"] = df_scaled["outlier"].apply(lambda x: 1 if x == -1 else 0)

    # Store results in PostgreSQL
    conn = psycopg2.connect(
        dbname="ann_test",
        user="postgres",
        password="123",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    
    for index, row in df.iterrows():
        cursor.execute(
            "UPDATE mytable SET is_outlier = %s WHERE id = %s",
            (df_scaled.loc[index, "outlier"], row["id"])
        )

    conn.commit()
    cursor.close()
    conn.close()

    return df_scaled
