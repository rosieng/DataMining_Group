import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression)
from sklearn.ensemble import (RandomForestRegressor)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

df = pd.read_csv(
    r"C:\Users\thanh\PycharmProjects\PythonProject2\DataMining_Group\listings.csv")

print(df.shape)
print(df.columns.tolist())

# 1.Business Question
# Which listing characteristics most strongly influence Airbnb prices in New York City?
df["price"] = (df["price"].replace("[$,]", "", regex=True).astype(float))

# 2.Data Cleaning & Feature Engineer
df["price"] = (df["price"]
    .replace("[$,]", "", regex=True)
    .astype(float))

df = df[df["price"] > 0]
df["log_price"] = np.log1p(df["price"])
df["professional_host"] = (df["host_listings_count"] >= 5).astype(int)
df["host_experience_years"] = (df["hosts_time_as_host_years"])
df["occupied_days"] = (365 - df["availability_365"])
# check bathroom
df["bathrooms_num"] = (df["bathrooms_text"].str.extract(r"(\d+\.?\d*)").astype(float))
df["bathrooms_num"] = (df["bathrooms_num"].fillna(df["bathrooms_num"].median()))

# 3.Feature selection
# why downsize from 90 var. to 15/16 vars
features = [
    "room_type",
    "property_type",
    "accommodates",
    "bathrooms_num",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "latitude",
    "longitude",
    "professional_host",
    "host_experience_years"]

# 4.Modelling data
model_df = df[features +["log_price"]].copy()

# 5.Missing value
model_df["reviews_per_month"] = (model_df["reviews_per_month"].fillna(0))
model_df["review_scores_rating"] = (model_df["review_scores_rating"].fillna(model_df["review_scores_rating"].median()))
model_df["bedrooms"] = (model_df["bedrooms"].fillna(model_df["bedrooms"].median()))
model_df["beds"] = (model_df["beds"].fillna(model_df["beds"].median()))
model_df["host_experience_years"] = (model_df["host_experience_years"].fillna(model_df["host_experience_years"].median()))

# 6.Check data
print(model_df.shape)

print(model_df.isnull()
    .sum()
    .sort_values(ascending=False))
model_df = model_df.dropna()

# 7.X & y
X = model_df[features]
y = model_df["log_price"]

# 8.One hot encoding
X = pd.get_dummies(X, drop_first=True)

# 9.Train Test list
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# 10. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 11.Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 12.Evalutaion
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# 13.Results
mae_lr, rmse_lr, r2_lr = (evaluate_model(y_test, y_pred_lr))
mae_rf, rmse_rf, r2_rf = (evaluate_model(y_test, y_pred_rf))
results = pd.DataFrame({
    "Model":["Linear Regression", "Random Forest"],
    "MAE":[mae_lr, mae_rf],
    "RMSE":[rmse_lr, rmse_rf],
    "R2":[r2_lr, r2_rf]
})
print(results)

# Train vs Test Performance
y_pred_train = rf.predict(X_train)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_rf)
print("Train R2:", train_r2)
print("Test R2:", test_r2)

# Random Forest: setup (hyperparameters)?

# Random Forest Importance Features
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
importance_df = (importance_df.sort_values("Importance", ascending=False))
print(importance_df.head(15).sort_values(ascending=False, by=["Importance"]))

# Cross validation : check if model is overfitted or not
# Test RMSE ~ CV RMSE
cv_rf = np.sqrt(
    -cross_val_score(
        rf,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )
)

print(cv_rf.mean())

# Residual Test
residuals = (y_test - y_pred_rf)
plt.figure(figsize=(8,5))

plt.scatter(y_pred_rf, residuals, alpha=0.3)
plt.axhline(y=0, color="red")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

df["minimum_nights"].describe()
df[["minimum_nights", "price"]].corr()
sns.scatterplot(data=df, x="minimum_nights", y="price")

cv_rf = np.sqrt(
    -cross_val_score(
        rf,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )
)

print(cv_rf.mean())
print(cv_rf.std())

