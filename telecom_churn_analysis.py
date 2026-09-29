"""
IBM Data Analytics with AI Internship Project
Project: Telecom Customer Churn Analysis & Retention Insights

Run:
    python telecom_churn_analysis.py

The script downloads the public IBM Telco Customer Churn sample dataset,
cleans it, performs EDA, trains a Logistic Regression model, saves charts,
and writes a machine-readable summary for the final report.
"""

from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"

def download_data():
    if DATA_FILE.exists():
        return
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

def load_and_clean():
    df = pd.read_csv(DATA_FILE)

    # Standardize whitespace in text columns.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # TotalCharges contains blank strings for some new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # Remove rows that cannot be used for the analysis.
    df = df.dropna(subset=["TotalCharges"]).drop_duplicates(subset=["customerID"])

    # Business-friendly derived fields.
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["ServiceCount"] = sum((df[c] == "Yes").astype(int) for c in service_cols)
    df["TenureBand"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
    )
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

    return df

def save_chart_churn_rate(df):
    rate = df["ChurnFlag"].mean() * 100
    plt.figure(figsize=(7, 5))
    plt.bar(["Retained", "Churned"],
            [100-rate, rate])
    plt.ylabel("Customers (%)")
    plt.title("Customer Churn Overview")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_churn_overview.png", dpi=180)
    plt.close()

def save_chart_contract(df):
    rates = df.groupby("Contract", observed=False)["ChurnFlag"].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(8, 5))
    rates.plot(kind="bar")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Contract Type")
    plt.title("Churn Rate by Contract Type")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_churn_by_contract.png", dpi=180)
    plt.close()

def save_chart_tenure(df):
    rates = df.groupby("TenureBand", observed=False)["ChurnFlag"].mean() * 100
    plt.figure(figsize=(8, 5))
    rates.plot(kind="bar")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Tenure Band")
    plt.title("Churn Rate by Customer Tenure")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_churn_by_tenure.png", dpi=180)
    plt.close()

def save_chart_payment(df):
    rates = df.groupby("PaymentMethod", observed=False)["ChurnFlag"].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(9, 5))
    rates.plot(kind="bar")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Payment Method")
    plt.title("Churn Rate by Payment Method")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_churn_by_payment.png", dpi=180)
    plt.close()

def train_model(df):
    features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]
    X = df[features]
    y = df["ChurnFlag"]

    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in features if c not in categorical]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, pred, output_dict=True)
    auc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)

    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "model_classification_report.csv")
    pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    with open(OUTPUT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Test records: {len(y_test)}\n")
        f.write("Target: Churn (1=Yes, 0=No)\n")

def create_summary(df):
    overall = df["ChurnFlag"].mean() * 100
    contract = (df.groupby("Contract")["ChurnFlag"].mean() * 100).sort_values(ascending=False)
    tenure = (df.groupby("TenureBand", observed=False)["ChurnFlag"].mean() * 100).sort_values(ascending=False)
    payment = (df.groupby("PaymentMethod")["ChurnFlag"].mean() * 100).sort_values(ascending=False)

    summary = pd.DataFrame({
        "Metric": [
            "Records after cleaning",
            "Overall churn rate",
            "Highest-churn contract",
            "Highest-churn contract rate",
            "Highest-churn tenure band",
            "Highest-churn tenure rate",
            "Highest-churn payment method",
            "Highest-churn payment rate"
        ],
        "Value": [
            len(df),
            f"{overall:.2f}%",
            contract.index[0],
            f"{contract.iloc[0]:.2f}%",
            str(tenure.index[0]),
            f"{tenure.iloc[0]:.2f}%",
            payment.index[0],
            f"{payment.iloc[0]:.2f}%"
        ]
    })
    summary.to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False)

def main():
    download_data()
    df = load_and_clean()
    df.to_csv(OUTPUT_DIR / "cleaned_telco_churn.csv", index=False)

    save_chart_churn_rate(df)
    save_chart_contract(df)
    save_chart_tenure(df)
    save_chart_payment(df)
    create_summary(df)
    train_model(df)

    print("\nProject completed successfully.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("Use analysis_summary.csv and the four PNG charts in the project report.")

if __name__ == "__main__":
    main()