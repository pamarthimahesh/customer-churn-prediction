import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict customer churn and analyze key insights using Machine Learning")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()

# ---------------- SPLIT DATA ----------------
X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🔧 Enter Customer Details")

credit_score = st.sidebar.slider("Credit Score", 350, 850, 600)
age = st.sidebar.slider("Age", 18, 90, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 50000.0)
products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
active = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# ---------------- INPUT ENCODING ----------------
input_data = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products,
    'HasCrCard': 1 if has_card == "Yes" else 0,
    'IsActiveMember': 1 if active == "Yes" else 0,
    'EstimatedSalary': salary,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0
}
# Create input dataframe
input_df = pd.DataFrame([input_data])

# Align input with training features
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)


# ---------------- PREDICTION ----------------
st.subheader("🔮 Prediction Result")

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer is LIKELY to churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\nProbability: {probability:.2f}")

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("📈 Model Performance")
y_pred = model.predict(X_test)
st.write("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- DASHBOARD CHARTS ----------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

# Churn Distribution
with col1:
    st.markdown("### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    ax.set_xlabel("Churn (0 = No, 1 = Yes)")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

# Gender vs Churn
with col2:
    st.markdown("### Churn by Gender")
    gender_churn = df.groupby('Gender_Male')['Exited'].mean()
    gender_churn.index = ['Female', 'Male']
    fig, ax = plt.subplots()
    gender_churn.plot(kind='bar', ax=ax)
    ax.set_ylabel("Churn Rate")
    st.pyplot(fig)

# Geography vs Churn
st.markdown("### 🌍 Churn by Geography")
geo_cols = [c for c in df.columns if "Geography_" in c]
geo_rates = {
    "France": df[df[geo_cols].sum(axis=1) == 0]['Exited'].mean(),
    "Germany": df[df['Geography_Germany'] == 1]['Exited'].mean(),
    "Spain": df[df['Geography_Spain'] == 1]['Exited'].mean()
}

fig, ax = plt.subplots()
ax.bar(geo_rates.keys(), geo_rates.values())
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# Feature Importance
st.markdown("### 🌲 Top Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
st.pyplot(fig)

# ---------------- BUSINESS INSIGHTS ----------------
st.subheader("📌 Key Business Insights")

st.markdown("""
- Customers with **low credit score and high balance** have higher churn risk  
- **Inactive members** are more likely to leave  
- **Germany** shows the highest churn rate among regions  
- **Age and balance** are strong churn indicators  
- Random Forest effectively captures complex churn patterns  
""")

st.success("🎉 End-to-End Customer Churn Prediction App Ready!")
