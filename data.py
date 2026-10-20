import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================
# 1. LOAD DATA
# ==========================

path = kagglehub.dataset_download(
    "dgomonov/new-york-city-airbnb-open-data"
)

print("Dataset path:", path)
print(os.listdir(path))

df = pd.read_csv(path + "\\AB_NYC_2019.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())


# ==========================
# 2. FEATURE ENGINEERING
# ==========================

# Occupied Days
df["occupied_days"] = 365 - df["availability_365"]

# Professional Host Flag
df["professional_host"] = (
    df["calculated_host_listings_count"] >= 5
).astype(int)

# Review Intensity
df["review_intensity"] = (
    df["number_of_reviews"]
    * df["reviews_per_month"].fillna(0)
)

# Log Price
df["log_price"] = np.log1p(df["price"])

print("\nFeature Engineering Complete")


# ==========================
# 3. MISSING VALUE ANALYSIS
# ==========================

missing_pct = (
    df.isnull().mean() * 100
).sort_values(ascending=False)

missing_only = missing_pct[missing_pct > 0]

print("\nMissing Values (%):")
print(missing_only)

if len(missing_only) > 0:
    plt.figure(figsize=(8, 5))

    missing_only.plot(kind="bar")

    plt.title("Percentage of Missing Values")
    plt.ylabel("% Missing")
    plt.tight_layout()

    plt.show()


# ==========================
# 4. CORRELATION ANALYSIS
# ==========================

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.show()


# ==========================
# 5. CORRELATION WITH LOG PRICE
# ==========================

price_corr = (
    corr_matrix["log_price"]
    .drop("log_price")
    .sort_values()
)

print("\nCorrelation with Log Price:")
print(price_corr)

plt.figure(figsize=(8, 6))

price_corr.plot(
    kind="barh"
)

plt.title("Correlation with Log Price")

plt.tight_layout()

plt.show()


# ==========================
# 6. PRICE DISTRIBUTION
# ==========================

df[
    ["price", "log_price"]
].hist(
    figsize=(10, 4)
)

plt.suptitle("Price vs Log(Price) Distribution")

plt.show()


# ==========================
# 7. ROOM TYPE vs PRICE
# ==========================

df_box = df[df["price"] < 1000]

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_box,
    x="room_type",
    y="price"
)

plt.title("Price by Room Type")

plt.show()


# ==========================
# 8. NEIGHBOURHOOD GROUP vs PRICE
# ==========================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_box,
    x="neighbourhood_group",
    y="price"
)

plt.title("Price by Neighbourhood Group")

plt.show()


# ==========================
# 9. QUICK SUMMARY TABLES
# ==========================

print("\nAverage Price by Room Type")
print(
    df.groupby("room_type")["price"]
      .mean()
      .sort_values(ascending=False)
)

print("\nAverage Price by Neighbourhood Group")
print(
    df.groupby("neighbourhood_group")["price"]
      .mean()
      .sort_values(ascending=False)
)

print("\nEDA Completed Successfully")

# ==========================
# 10. DATA CLEANING
# ==========================

# reviews_per_month
df["reviews_per_month"] = (
    df["reviews_per_month"]
    .fillna(0)
)

# remove rows with missing price
df = df.dropna(subset=["price"])

print("\nDataset shape after cleaning:")
print(df.shape)

# ==========================
# 11. FEATURE SELECTION
# ==========================

features = [
    "room_type",
    "neighbourhood_group",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "occupied_days",
    "professional_host",
    "review_intensity",
    "latitude",
    "longitude"
]

target = "log_price"

# ==========================
# 12. TRAIN TEST SPLIT
# ==========================

from sklearn.model_selection import train_test_split

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==========================
# 13. ENCODING
# ==========================

X_train = pd.get_dummies(
    X_train,
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    drop_first=True
)

X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

print("\nEncoded columns:")
print(X_train.columns)

# ==========================
# 14. LINEAR REGRESSION
# ==========================

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)

y_pred_lr = lr.predict(
    X_test
)

# ==========================
# 15. EVALUATE LR
# ==========================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae_lr = mean_absolute_error(
    y_test,
    y_pred_lr
)

rmse_lr = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_lr
    )
)

r2_lr = r2_score(
    y_test,
    y_pred_lr
)

print("\nLinear Regression Results")
print("MAE :", round(mae_lr,4))
print("RMSE:", round(rmse_lr,4))
print("R²  :", round(r2_lr,4))

# ==========================
# 16. LASSO
# ==========================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

lasso = Lasso(
    alpha=0.01
)

lasso.fit(
    X_train_scaled,
    y_train
)

y_pred_lasso = lasso.predict(
    X_test_scaled
)

# ==========================
# 17. EVALUATE LASSO
# ==========================

mae_lasso = mean_absolute_error(
    y_test,
    y_pred_lasso
)

rmse_lasso = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_lasso
    )
)

r2_lasso = r2_score(
    y_test,
    y_pred_lasso
)

print("\nLasso Results")
print("MAE :", round(mae_lasso,4))
print("RMSE:", round(rmse_lasso,4))
print("R²  :", round(r2_lasso,4))

# ==========================
# 18. MODEL COMPARISON
# ==========================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Lasso"
    ],
    "MAE": [
        mae_lr,
        mae_lasso
    ],
    "RMSE": [
        rmse_lr,
        rmse_lasso
    ],
    "R2": [
        r2_lr,
        r2_lasso
    ]
})

print("\nModel Comparison")
print(results)

# ==========================
# 19. LR COEFFICIENTS
# ==========================

coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lr.coef_
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    key=lambda x: abs(x),
    ascending=False
)

print("\nTop LR Coefficients")
print(coef_df.head(15))

# ==========================
# 20. LASSO FEATURES
# ==========================

lasso_coef = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lasso.coef_
})

lasso_coef = lasso_coef[
    lasso_coef["Coefficient"] != 0
]

lasso_coef = lasso_coef.sort_values(
    by="Coefficient",
    key=lambda x: abs(x),
    ascending=False
)

print("\nFeatures Kept By Lasso")
print(lasso_coef)

# ==========================
# 18. DECISION TREE
# ==========================

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree.fit(
    X_train,
    y_train
)

y_pred_tree = tree.predict(
    X_test
)

# ==========================
# 19. EVALUATE TREE
# ==========================

mae_tree = mean_absolute_error(
    y_test,
    y_pred_tree
)

rmse_tree = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_tree
    )
)

r2_tree = r2_score(
    y_test,
    y_pred_tree
)

print("\nDecision Tree Results")
print("MAE :", round(mae_tree,4))
print("RMSE:", round(rmse_tree,4))
print("R²  :", round(r2_tree,4))

# ==========================
# MODEL COMPARISON
# ==========================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Lasso",
        "Decision Tree"
    ],
    "MAE": [
        mae_lr,
        mae_lasso,
        mae_tree
    ],
    "RMSE": [
        rmse_lr,
        rmse_lasso,
        rmse_tree
    ],
    "R2": [
        r2_lr,
        r2_lasso,
        r2_tree
    ]
})

print(results)

# ==========================
# TREE FEATURE IMPORTANCE
# ==========================

tree_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": tree.feature_importances_})

tree_importance = tree_importance.sort_values(
    by="Importance",
    ascending=False)

print("\nDecision Tree Feature Importance")
print(tree_importance.head(15))

plt.figure(figsize=(8,6))

sns.barplot(
    data=tree_importance.head(10),
    x="Importance",
    y="Feature")

plt.title("Decision Tree Feature Importance")
plt.show()

