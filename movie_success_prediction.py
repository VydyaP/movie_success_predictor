# movie_success_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
)

# 1. Load Dataset
df = pd.read_csv("imdb_top_1000.csv")

# 2. Replace missing values
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# 3. Clean Gross Column
df["Gross"] = df["Gross"].replace('[\$,]', '', regex=True).replace(',', '', regex=True)
df["Gross"] = pd.to_numeric(df["Gross"], errors='coerce')
df.dropna(subset=["Gross"], inplace=True)

# 4. Create Target Label
df["Gross_Label"] = df["Gross"].apply(lambda x: "Success" if x > 50000000 else "Flop")

# 5. Filter and Sample for Balancing
success_df = df[df["Gross_Label"] == "Success"].sample(n=80, random_state=42)
flop_df = df[df["Gross_Label"] == "Flop"].sample(n=20, random_state=42)
balanced_df = pd.concat([success_df, flop_df])

# 6. Features and Labels
features = ["IMDB_Rating", "Meta_score", "No_of_Votes", "Runtime"]  # Choose your numeric columns
balanced_df["Runtime"] = balanced_df["Runtime"].str.replace(" min", "").astype(float)
X = balanced_df[features]
y = balanced_df["Gross_Label"]

# 7. Encode Label
y = y.map({"Success": 1, "Flop": 0})

# 8. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 9. Train Model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 10. Predict & Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc * 100)
print("Kappa:", kappa)
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Flop", "Success"]))
