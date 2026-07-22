import pandas as pd

df = pd.read_csv(r"C:\Users\megha\OneDrive\Certificates\diabetes.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInformation:")
print(df.info())

print("\nStatistics:")
print(df.describe())

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(predictions[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(accuracy)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, pred_lr))

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, pred_dt))

preg = float(input("Enter Pregnancies: "))
glu = float(input("Enter Glucose: "))
bp = float(input("Enter Blood Pressure: "))
skin = float(input("Enter Skin Thickness: "))
insulin = float(input("Enter Insulin: "))
bmi = float(input("Enter BMI: "))
dpf = float(input("Enter Diabetes Pedigree Function: "))
age = float(input("Enter Age: "))

sample = pd.DataFrame(
    [[preg, glu, bp, skin, insulin, bmi, dpf, age]],
    columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]
)


result = lr.predict(sample)

if result[0] == 1:
    print("Patient is likely Diabetic")
else:
    print("Patient is likely Non-Diabetic")

import matplotlib.pyplot as plt

algorithms = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracy = [74, 74, 72]

plt.bar(algorithms, accuracy)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.show()

import matplotlib.pyplot as plt

df['Outcome'].value_counts().plot(kind='bar')

plt.title("Diabetes Distribution")
plt.xlabel("Outcome")
plt.ylabel("Count")

plt.show()