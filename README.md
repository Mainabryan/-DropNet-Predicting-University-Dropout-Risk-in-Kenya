##OVERVIEW OF STREAMLIT 
![Screenshot](https://github.com/Mainabryan/-DropNet-Predicting-University-Dropout-Risk-in-Kenya/blob/93ff62d28d49ad56af338f04251cc15b4e2e9db0/Screenshot%202025-07-28%20201059.png)


# -DropNet-Predicting-University-Dropout-Risk-in-Kenya
Predicts university dropout risk in Kenya using logistic regression based on household income, HELB support, and financial stress levels. Built with Python and Streamlit.
# 🎓 Kenya University Dropout Risk Predictor

A data-driven machine learning project that predicts the likelihood of a university student in Kenya dropping out due to financial stress, limited HELB support, or high academic burden.

---

## 📌 Problem Overview

Many Kenyan university students face **financial instability**, rising tuition fees, reduced government support (like HELB), and mental health struggles. These factors directly impact student retention, performance, and future opportunities.

With recent government changes in student funding, it's become crucial to understand **which students are most at risk of dropping out** — and why.

---

## ✅ Project Goal

To build an **interactive machine learning app** that:
- Predicts dropout risk based on a student’s background and financial data
- Highlights the impact of financial aid, scholarships, and mental health risk
- Offers policymakers and citizens a way to simulate funding outcomes

---)
## 📥 1. Loading the Dataset & EDA (Exploratory Data Analysis)

We began by importing a custom dataset simulating the financial and academic details of university students in Kenya.

### Why This Step Matters:
- Understanding the **structure and types of data** is critical before modeling.
- We explored key variables like `household_income`, `program_cost_per_year`, and `financial_stress_level` to identify trends, outliers, and relationships.

### Actions Taken:
- Loaded the data using Pandas
- Inspected data types, column distribution, and sample rows
- Visualized variables using **Seaborn** and **Matplotlib**:
  - Histograms for income & academic performance
  - Count plots for dropout risk by stress level
  - Box plots to check income vs dropout patterns

This step helped us frame the **financial vulnerability** among students and detect early signs of imbalance or skew.


![screenshot](https://github.com/Mainabryan/-DropNet-Predicting-University-Dropout-Risk-in-Kenya/blob/63e644752b27675a56c3f807c369cefa975055e4/Screenshot%202025-07-16%20121017.png)
![SCREENSHOT](https://github.com/Mainabryan/-DropNet-Predicting-University-Dropout-Risk-in-Kenya/blob/d8814c1740d56dd39cdacb808f200946818362d4/Screenshot%202025-07-16%20121034.png)

## 🚨 2. Handling Missing Values

Like most real-world datasets, some columns had missing values (e.g., `helb_amount`, `scholarship_amount`, and `mental_health_risk`).

### Why This Step Matters:
- Models can't learn from incomplete data
- Naively removing rows can lead to bias or data loss

### Actions Taken:
- Used **median** for numeric columns with outliers (`household_income`, `program_cost_per_year`)
- Used **0** for support-based features like `helb_amount`, `scholarship_amount`, and `other_support` — assuming no support
- Used **mode** for categorical features like `financial_stress_level` and `mental_health_risk`

This preserved the dataset's size and kept the structure realistic without injecting false assumptions.

![SCREENSHOT](https://github.com/Mainabryan/-DropNet-Predicting-University-Dropout-Risk-in-Kenya/blob/23f931110e295fc822d7d8c4c15e8666f57b65ab/Screenshot%202025-07-28%20194823.png)

## 🧩 3. Encoding Categorical Variables

The dataset included several **categorical features** such as:

- `county` (e.g., Nairobi, Kisumu, Mombasa)
- `financial_stress_level` (Low, Medium, High)
- `mental_health_risk` (Low, Medium, High)
- `public_uni` (Yes/No)

### Why This Step Matters:
Machine learning models like logistic regression require **numerical input**, so categorical features must be encoded in a way the model can interpret — without losing their meaning.

### Encoding Strategy:
- For **ordinal variables** like `financial_stress_level` and `mental_health_risk`, we used **Label Encoding**:
  - Low → 0, Medium → 1, High → 2
- For **nominal variables** like `county`, we also used **Label Encoding** to keep the feature lightweight
- Converted `public_uni` from Yes/No to binary 1/0

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Ordinal encoding
data['financial_stress_level'] = le.fit_transform(data['financial_stress_level'])
data['mental_health_risk'] = le.fit_transform(data['mental_health_risk'])

# Nominal encoding
data['county'] = le.fit_transform(data['county'])
data['public_uni'] = data['public_uni'].map({'Yes': 1, 'No': 0})

## 🧠 ML Model Use)

We trained a **Logistic Regression model** using synthetic but realistic student data to predict:
> Whether a student is likely to drop out (`1`) or not (`0`)

The model considers:
- Household income
- HELB and scholarship support
- Program cost
- Stress level and affordability basics

> The model is deployed through a fully interactive **Streamlit app** that lets users explore the dataset, visualize patterns, and make predictions based on input sliders and dropdowns.

---

## 🛠️ Tools & Technologies

| Tool           | Purpose                          |
|----------------|----------------------------------|
| Python         | Core programming language        |
| Pandas         | Data cleaning and manipulation   |
| Scikit-learn   | Logistic Regression + Preprocessing |
| Matplotlib/Seaborn | Visualizations                |
| Streamlit      | Interactive app UI               |
| GitHub         | Version control and collaboration |

---
## 📊 4. Feature Scaling

To prepare the dataset for logistic regression, we applied **feature scaling** to normalize numeric columns like:

- `household_income`
- `program_cost_per_year`
- `helb_amount`
- `scholarship_amount`
- `other_support`

### Why This Step Matters:
Logistic Regression is sensitive to feature magnitudes. Without scaling, features with large values (e.g., `household_income`) could **dominate** others (e.g., `helb_amount`), resulting in a biased model.

### Method Used:
We used **StandardScaler** from `scikit-learn`, which transforms data so that:
- Mean = 0
- Standard deviation = 1

### Code Snippet:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Apply scaling to relevant columns
cols_to_scale = ['household_income', 'program_cost_per_year', 'helb_amount', 
                 'scholarship_amount', 'other_support']

data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

## 🤖 5. Model Training & Evaluation

With a clean, encoded, and scaled dataset, we trained a **Logistic Regression model** to predict the likelihood of a university student dropping out.

---

### 🧪 Train-Test Split

We split the dataset into:

- **80% training data** – for model learning
- **20% test data** – for evaluation

```python
from sklearn.model_selection import train_test_split

X = data.drop("likely_to_dropout", axis=1)
y = data["likely_to_dropout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


