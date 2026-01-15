import numpy as np
import pandas as pd
from DataSanitize import prepare_data
from LogisticRegression import MyLogisticRegression, calculate_metrics
from NaiveBayes import CustomNaiveBayes
from DecisionTree import SimpleID3

# 1. Pregatirea Datelor
FILE_NAME = 'ap_dataset.csv'
TARGET_COL = 'Crazy Sauce'

try:
    df_final = prepare_data(FILE_NAME)
    # doar bonurile cu Crazy Schnitzel
    df_task = df_final[df_final['Crazy Schnitzel'] > 0].copy()

    # Pregatire X și y
    y = (df_task[TARGET_COL] > 0).astype(int).values
    cols_to_drop = [TARGET_COL, 'Crazy Schnitzel', 'data_bon']
    X_df = df_task.drop(columns=[c for c in cols_to_drop if c in df_task.columns])
    X = X_df.values

    # Scalare (pt Regresia Logistică)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / (std + 1e-8)

    # Split 80/20
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

except Exception as e:
    print(f"Eroare la procesarea datelor: {e}")
    exit()

print(f"Date incarcate cu succes. Testare pe {len(X_test)} eșantioane.\n")

# --- COMPARARE MODELE ---

# 1. Antrenare Regresie Logistica
print("Antrenare Regresie Logistica...")
lr = MyLogisticRegression(learning_rate=0.1, num_iterations=2000)
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)

# 2. Antrenare Naive Bayes (Necesita valori non-negative)
print("Antrenare Naive Bayes...")
nb = CustomNaiveBayes()
# Shiftam datele pentru a fi pozitive (necesar pentru Multinomial NB)
min_val = X_train.min()
X_train_nb = X_train - min_val
X_test_nb = X_test - min_val
nb.fit(X_train_nb, y_train)
preds_nb = nb.predict(X_test_nb)

# 3. Antrenare ID3 Decision Tree
print("Antrenare Decision Tree...")
dt = SimpleID3(depth_limit=6)
dt.fit(X_train, y_train)
preds_dt = dt.predict(X_test)

# --- REZULTATE FINALE ---
print("\n" + "=" * 30)
print("   REZULTATE COMPARATIVE")
print("=" * 30)

calculate_metrics(y_test, preds_lr, "REGRESIE LOGISTICA")
calculate_metrics(y_test, preds_nb, "NAIVE BAYES")
calculate_metrics(y_test, preds_dt, "DECISION TREE (ID3)")