import pandas as pd 
import joblib

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

print("Model Loaded Successfully!")

test_df = pd.read_csv("test.csv")

test_ids = test_df["Id"]

print(test_df.head())
print(test_df.shape)
print(test_df.columns.tolist())


test_df = test_df.drop(
    columns=[
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu"
    ]
)

print(test_df.shape)

test_df["LotFrontage"] = test_df["LotFrontage"].fillna(test_df["LotFrontage"].median())
test_df["GarageYrBlt"] = test_df["GarageYrBlt"].fillna(test_df["GarageYrBlt"].median())
test_df["MasVnrArea"] = test_df["MasVnrArea"].fillna(test_df["MasVnrArea"].median())

garage_cols = ["GarageQual", "GarageType", "GarageCond", "GarageFinish"]
bsmt_cols = ["BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtExposure"]


for col in garage_cols:
    test_df[col] = test_df[col].fillna("None")

for col in bsmt_cols:
    test_df[col] = test_df[col].fillna("None")

test_df["Electrical"] = test_df["Electrical"].fillna(test_df["Electrical"].mode()[0])

test_df["BsmtHalfBath"] = test_df["BsmtHalfBath"].fillna(test_df["BsmtHalfBath"].median())
test_df["BsmtFullBath"] = test_df["BsmtFullBath"].fillna(test_df["BsmtFullBath"].median())
test_df["TotalBsmtSF"] = test_df["TotalBsmtSF"].fillna(test_df["TotalBsmtSF"].median())
test_df["BsmtUnfSF"] = test_df["BsmtUnfSF"].fillna(test_df["BsmtUnfSF"].median())
test_df["GarageArea"] = test_df["GarageArea"].fillna(test_df["GarageArea"].median())
test_df["BsmtFinSF1"] = test_df["BsmtFinSF1"].fillna(test_df["BsmtFinSF1"].median())
test_df["BsmtFinSF2"] = test_df["BsmtFinSF2"].fillna(test_df["BsmtFinSF2"].median())
test_df["GarageCars"] = test_df["GarageCars"].fillna(test_df["GarageCars"].median())

test_df = pd.get_dummies(test_df)
test_df = test_df.reindex(columns=model_columns, fill_value=0)

print(test_df.shape)

# print(test_df.isnull().sum().sum())

predictions = model.predict(test_df)

print(predictions[:10])

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

