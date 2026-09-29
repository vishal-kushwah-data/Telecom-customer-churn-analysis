# Telecom Customer Churn Analysis & Retention Insights

## IBM Data Analytics with AI Internship Project

### 1. Project Overview
This project analyzes customer churn for a fictional telecommunications company using the public IBM Telco Customer Churn sample dataset.

The objective is to convert customer-level data into business insights:
**Data → Information → Insights → Decision → Action**

The analysis focuses on:
- overall churn
- contract type
- customer tenure
- payment method
- service usage
- customer retention opportunities
- an optional Logistic Regression churn classification model

### 2. Problem Statement
Telecom businesses need to understand why customers leave and which customer segments show higher churn. This project identifies churn patterns and creates a simple analytical framework that can support targeted retention activities.

### 3. Dataset
**Source:** IBM Telco Customer Churn sample dataset

**Direct dataset URL:**
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The dataset contains customer demographics, subscribed services, account information, charges, contract/payment details, and a `Churn` outcome.

The IBM repository is archived but remains publicly accessible. The dataset is described as sample data for a fictional telecommunications company.

### 4. Technology Stack
- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn

### 5. Project Workflow
1. Download the dataset.
2. Clean text and numeric fields.
3. Convert `TotalCharges` to numeric.
4. Remove unusable rows and duplicate customer IDs.
5. Create derived fields such as `ServiceCount`, `TenureBand`, and `ChurnFlag`.
6. Perform exploratory analysis.
7. Generate four business charts.
8. Train a Logistic Regression model.
9. Save model evaluation results.
10. Save a final analysis summary.

### 6. Files Generated After Running
The script creates an `outputs/` folder containing:
- `cleaned_telco_churn.csv`
- `analysis_summary.csv`
- `01_churn_overview.png`
- `02_churn_by_contract.png`
- `03_churn_by_tenure.png`
- `04_churn_by_payment.png`
- `model_classification_report.csv`
- `confusion_matrix.csv`
- `model_summary.txt`

### 7. How to Run
Install Python 3.10+.

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
python telecom_churn_analysis.py
```

The script downloads the dataset automatically and generates the analysis outputs.

### 8. Business Interpretation
The project follows the framework taught in the internship masterclass:
- KPIs show what is happening.
- Trends show direction.
- Drivers help explain patterns.
- Risks/opportunities identify where attention is needed.
- Recommended actions connect analysis to business decisions.

### 9. Important Limitation
This is a cross-sectional sample dataset. The churn label represents whether the customer left within the observed period; the model should not be described as proof that a particular retention action causes a customer to stay.

### 10. Academic/Portfolio Note
This project is for educational and portfolio purposes. The dataset is credited to IBM's public sample data.

## Author
Ashi Kushwah
B.Tech CSE
Vikrant University, Gwalior