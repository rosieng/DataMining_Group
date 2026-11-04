# 1.Import Packages
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score)

# 2.Load Data
path = kagglehub.dataset_download("dgomonov/new-york-city-airbnb-open-data")
df = pd.read_csv(path + "\\AB_NYC_2019.csv")

print(df.shape)
print(df.head())
print(df.info())

# 3.Feature Engineering
df["occupied_days"] = (365 - df["availability_365"])
df["professional_host"] = (df["calculated_host_listings_count"] >= 5).astype(int)
df["review_intensity"] = (df["number_of_reviews"] * df["reviews_per_month"].fillna(0))
df["log_price"] = np.log1p(df["price"])

# 4.Missing Value
missing = (df.isnull().mean().sort_values(ascending=False) * 100)
print(missing)

# 5.Data Cleaning
df["reviews_per_month"] = (df["reviews_per_month"].fillna(0))
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]
df = df[df["price"] < 1000]
print(df.shape)

# 6.Univariate Analysis
# Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["price"], bins=50)
plt.show()

# Log Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["log_price"], bins=50)
plt.show()

# 7.Bivariate Analysis
# Room Type vs Price
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="room_type", y="price")
plt.show()

# Area vs Price
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="neighbourhood_group",y="price")
plt.show()

# 8.Correlation Matrix
numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()
plt.figure(figsize=(12,8))

sns.heatmap(corr, cmap="coolwarm",center=0)
plt.show()

#9. Average Price Tables
print(df.groupby("room_type")["price"]
     .mean()
     .sort_values(ascending=False))

print(df.groupby("neighbourhood_group")["price"]
     .mean()
     .sort_values(ascending=False))

#10. Prepare Features
features = ["room_type", "neighbourhood_group", "minimum_nights", "number_of_reviews", "reviews_per_month",
            "calculated_host_listings_count", "occupied_days", "professional_host", "review_intensity",
            "latitude", "longitude"]
target = "log_price"

#11. Train Test Split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = (train_test_split(
        X, y, test_size=0.20, random_state=42))

# 12.One-hot Encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = (X_train.align(X_test, join="left", axis=1, fill_value=0))

# 13.Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = (lr.predict(X_test))

# 14.Lasso
scaler = StandardScaler()
X_train_scaled = (scaler.fit_transform(X_train))
X_test_scaled = (scaler.transform(X_test))
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled,y_train)
y_pred_lasso = (lasso.predict(X_test_scaled))

# 15.Decision Tree
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = (tree.predict(X_test))

# 16.Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(mae_rf)
print(rmse_rf)
print(r2_rf)

# 17.Evalution
def evaluate_model(y_true,y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# 18.Evaluation Models
mae_lr, rmse_lr, r2_lr = (evaluate_model(y_test, y_pred_lr))
mae_lasso, rmse_lasso, r2_lasso = (evaluate_model(y_test, y_pred_lasso))
mae_tree, rmse_tree, r2_tree = (evaluate_model(y_test, y_pred_tree))
mae_rf, rmse_rf, r2_rf = (evaluate_model(y_test, y_pred_rf))

# 19.Compare Models
results = pd.DataFrame({
    "Model":["Linear Regression", "Lasso", "Decision Tree", "Random Forest"],
    "MAE":[mae_lr, mae_lasso, mae_tree, mae_rf],
    "RMSE":[rmse_lr, rmse_lasso, rmse_tree, rmse_rf],
    "R2":[r2_lr, r2_lasso, r2_tree, r2_rf]})
print(results)

# 20.Random Forest Feature Importance
rf_importance = pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
rf_importance = (rf_importance.sort_values("Importance", ascending=False))
print(rf_importance.head(15))

plt.figure(figsize=(8,6))
sns.barplot(data=rf_importance.head(10), x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.show()

# 21.Residual Analysis
residuals = (y_test - y_pred_rf)
plt.figure(figsize=(7,5))
plt.scatter(y_pred_rf, residuals, alpha=0.3)
plt.axhline(y=0, color="red")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 22.K-meam clustering
cluster_vars = ["occupied_days", "minimum_nights", "number_of_reviews", "review_intensity"]
scaler = StandardScaler()
cluster_data = (scaler.fit_transform(df[cluster_vars]))

# 23.Elbow method
inertia = []
for k in range(1,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(cluster_data)
    inertia.append(km.inertia_)
plt.plot(range(1,11), inertia,marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 24.Silhouette Score
for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(cluster_data)
    score = silhouette_score(cluster_data, labels)
    print(f"K={k} -> {score}")

# 25.Final K-mean
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = (kmeans.fit_predict(cluster_data))

# 26.Cluster profile
cluster_profile = (df.groupby("cluster").agg({
        "price":"mean",
        "occupied_days":"mean",
        "minimum_nights":"mean",
        "number_of_reviews":"mean",
        "review_intensity":"mean"})
)
print(cluster_profile.round(2))
