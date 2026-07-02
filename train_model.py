import joblib
import pandas as pd 

df = pd.read_csv("train.csv")

print(df.head())

print("\Dataset Shape: ", df.shape)

print("\nColumns: ", df.columns.to_list())

print("\nMissing Values: ")
print(df.isnull().sum().sort_values(ascending=False).head(20))

df =df.drop(columns=[
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu"
])

print(df.shape)
print(df.isnull().sum().sort_values(ascending=False).head(15))

df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())

garage_cols = ["GarageQual", "GarageType", "GarageCond", "GarageFinish"]
bsmt_cols = ["BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtExposure"]


for col in garage_cols:
    df[col] = df[col].fillna("None")

for col in bsmt_cols:
    df[col] = df[col].fillna("None")

df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

print(df.isnull().sum().sum())
print(df.isnull().sum().sort_values(ascending=False).head(10))

print(df["SalePrice"].describe())

correlation = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

print(correlation.head(15))


df = pd.get_dummies(df,drop_first=True)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("Feature Shape: ", X.shape)
print("Target Shape: ", y.shape)

model_columns = X.columns


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data: ", X_train.shape )
print("Testing Data: ", X_test.shape )


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

print("First 10 Predictions: ")
print(y_pred[:10])


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R² Score: ", r2)

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.round(2)
})

comparison["Error"] = (comparison["Actual Price"] - comparison["Predicted Price"]).round(2)

print(comparison.head(10))


print("Best Predictions")
print(comparison.sort_values(by="Error", key=abs).head(10))


print("Worst Predictions")
print(comparison.sort_values(by="Error", key=abs, ascending=False).head(10))

print(comparison.describe())


import matplotlib.pyplot as plt 
plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth = 2,
    label = "Perfect Prediction"
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.legend()

plt.savefig("Actual vs Predicted House Prices.png")
plt.show()

print(len(model_columns))

new_house = pd.DataFrame(0, index=[0], columns= model_columns)

joblib.dump(model, "house_price_model.pkl")
joblib.dump(model_columns, "model_columns.pkl")

print("Model and columns saved successfully!")

# print(new_house.head())
# print(new_house.shape)

# new_house["OverallQual"] = 8
# new_house["GrLivArea"] = 2200
# new_house["GarageCars"] = 2
# print(new_house[["OverallQual", "GrLivArea", "GarageCars"]])

overall_qual = int(input("Enter Overall Quality (1-10): "))
grlivarea = float(input("Enter Living Area (sq ft): "))
garage_cars = int(input("Enter Number of Garage Cars: "))
year_built = int(input("Enter Year Built: "))


new_house["OverallQual"] = overall_qual
new_house["GrLivArea"] = grlivarea
new_house["GarageCars"] = garage_cars
new_house["YearBuilt"] = year_built


predicted_price = model.predict(new_house)

print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")





