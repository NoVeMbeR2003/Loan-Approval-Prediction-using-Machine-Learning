import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("loan_approval_dataset.csv")


df.columns = df.columns.str.strip()


df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()


le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_loan_status = LabelEncoder()

df['education'] = le_education.fit_transform(df['education'])
df['self_employed'] = le_self_employed.fit_transform(df['self_employed'])
df['loan_status'] = le_loan_status.fit_transform(df['loan_status'])


X = df.drop(['loan_status', 'loan_id'], axis=1)
y = df['loan_status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = DecisionTreeClassifier(criterion='gini', max_depth=5)  # prevent overfitting
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


print("\n--- Enter Loan Details ---")

education_input = input("Education (Graduate/Not Graduate): ").strip().title()
self_emp_input = input("Self Employed (Yes/No): ").strip().title()


if education_input not in le_education.classes_:
    print("Invalid Education Input!")
    exit()

if self_emp_input not in le_self_employed.classes_:
    print("Invalid Self Employment Input!")
    exit()

user_input = {
    "no_of_dependents": int(input("Number of dependents: ")),
    "education": education_input,
    "self_employed": self_emp_input,
    "income_annum": float(input("Annual Income: ")),
    "loan_amount": float(input("Loan Amount: ")),
    "loan_term": int(input("Loan Term (in months): ")),
    "cibil_score": int(input("CIBIL Score: ")),
    "residential_assets_value": float(input("Residential Assets Value: ")),
    "commercial_assets_value": float(input("Commercial Assets Value: ")),
    "luxury_assets_value": float(input("Luxury Assets Value: ")),
    "bank_asset_value": float(input("Bank Asset Value: "))
}


user_df = pd.DataFrame([user_input])


user_df['education'] = le_education.transform(user_df['education'])
user_df['self_employed'] = le_self_employed.transform(user_df['self_employed'])


user_df = user_df[X.columns]


user_prediction = model.predict(user_df)


result = le_loan_status.inverse_transform(user_prediction)

print("\n🔍 Loan Status Prediction:", result[0])