# ============================================================
# Module 9 Assignment — Regression Models on Housing Dataset
# ============================================================

# ── Step 1: Import Libraries ──────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Step 2: Load the Dataset ──────────────────────────────────
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"

col_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# sep='\s+' means "one or more whitespace" (এই dataset এ space দিয়ে separated)
df = pd.read_csv(data_url, sep=r'\s+', header=None, names=col_names)
a
print("✅ Dataset Loaded Successfully!")
print(f"   Shape: {df.shape}  →  {df.shape[0]} rows, {df.shape[1]} columns")
print()
print("First 5 rows:")
print(df.head())
print()
print("Basic statistics:")
print(df.describe().round(2))

# ── Step 3: Split Features and Target ────────────────────────
X = df.drop('MEDV', axis=1)   # Features (সব column MEDV ছাড়া)
y = df['MEDV']                  # Target (house price)

# 80% train, 20% test | random_state=42 মানে প্রতিবার same split হবে
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Dataset Split Done!")
print(f"   Training samples  : {X_train.shape[0]}")
print(f"   Testing  samples  : {X_test.shape[0]}")

# ── Step 4: Train Models ──────────────────────────────────────
# Dictionary এ তিনটা model একসাথে রাখলাম — loop এ সব train হবে
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression' : Ridge(alpha=1.0),   # alpha = regularization strength
    'Lasso Regression' : Lasso(alpha=1.0)
}

# ── Step 5: Evaluate Models ───────────────────────────────────
results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)                          # RMSE = sqrt(MSE)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    results.append({
        'Model': name,
        'MSE'  : round(mse,  4),
        'RMSE' : round(rmse, 4),
        'R2'   : round(r2,   4),
        'MAE'  : round(mae,  4)
    })

# ── Step 6: Comparison Table ──────────────────────────────────
results_df = pd.DataFrame(results)

print("\n" + "="*62)
print("         📊 Model Comparison Table")
print("="*62)
print(results_df.to_string(index=False))
print("="*62)

# ── Step 7: Best Model Analysis ──────────────────────────────
best_r2  = results_df.loc[results_df['R2'].idxmax(),  'Model']
best_mse = results_df.loc[results_df['MSE'].idxmin(), 'Model']
best_mae = results_df.loc[results_df['MAE'].idxmin(), 'Model']

print(f"\n📌 Best R2  Score  → {best_r2}")
print(f"📌 Lowest MSE      → {best_mse}")
print(f"📌 Lowest MAE      → {best_mae}")

# ── What the metrics mean ─────────────────────────────────────
print("""
📖 Metric Explanation:
  MSE  (Mean Squared Error)      — error এর square এর average। কম হলে ভালো।
  RMSE (Root Mean Squared Error) — MSE এর square root। actual unit এ error দেখায়।
  R2   (R-squared)               — model কতটা variance explain করতে পারছে (1 = perfect)।
  MAE  (Mean Absolute Error)     — actual error এর average। কম হলে ভালো।

💡 Conclusion:
  তিনটা model এর result প্রায় কাছাকাছি।
  Linear Regression সামান্য ভালো perform করেছে (R2 = 0.6688)।
  Ridge ও Lasso regularization add করে overfitting কমায়,
  কিন্তু এই dataset এ সেটার বড় পার্থক্য দেখা যাচ্ছে না।
""")
