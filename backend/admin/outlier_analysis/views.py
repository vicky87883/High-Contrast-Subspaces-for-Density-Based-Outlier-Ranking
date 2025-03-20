from django.shortcuts import render

# Create your views here.
import pandas as pd
from sklearn.preprocessing import StandardScaler
import psycopg2
from django.http import JsonResponse
from .models import ThyroidCancerData

def fetch_data_from_db():
    """Fetch data from PostgreSQL"""
    conn = psycopg2.connect(
        dbname="ann_test",
        user="postgres",
        password="123",
        host="localhost",
        port="5432"
    )
    query = "SELECT * FROM mytable"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    """Handle missing values and normalize"""
    df.fillna(df.median(), inplace=True)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=['id'])), columns=df.columns[1:])
    return df_scaled
