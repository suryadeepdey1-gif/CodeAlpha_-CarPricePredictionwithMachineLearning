import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("car data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# -------------------------------
# Step 2: Check Missing Values
# -------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values if any
df.dropna(inplace=True)

# -------------------------------
# Step 3: Feature Engineering
# -------------------------------
df["Car_Age"] = 2026 - df["Year"]
df.drop("Year", axis=1, inplace=True)

# -------------------------------
# Step 4: Encode Categorical Data
# -------------------------------
le_car = LabelEncoder()
le_fuel = LabelEncoder()
le_sell = LabelEncoder()
le_trans = LabelEncoder()

df["Car_Name"] = le_car.fit_transform(df["Car_Name"])
df["Fuel_Type"] = le_fuel.fit_transform(df["Fuel_Type"])
df["Selling_type"] = le_sell.fit_transform(df["Selling_type"])
df["Transmission"] = le_trans.fit_transform(df["Transmission"])

# -------------------------------
# Step 5: Prepare Features
# -------------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeatures used for training:")
print(X.columns.tolist())

# -------------------------------
# Step 6: Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Step 7: Train Model
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Step 8: Prediction
# -------------------------------
y_pred = model.predict(X_test)

prediction = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices")
print(prediction.head(10))

# -------------------------------
# Step 9: Model Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R² Score:", round(r2,2))

# -------------------------------
# Step 10: Actual vs Predicted Plot
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Price")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.show()

# -------------------------------
# Step 11: Feature Importance
# -------------------------------
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)

plt.title("Feature Importance")
plt.show()

# -------------------------------
# Step 12: Predict Future Selling Price
# -------------------------------

print("\n========== Future Car Price Prediction ==========")

car_name = input("Enter Car Name: ").strip()

# Check if car name exists
if car_name not in le_car.classes_:
    print("\nCar name not found in dataset!")
    print("Available car names:")
    print(", ".join(sorted(le_car.classes_)))
    exit()

present_price = float(input("Enter Current Showroom Price (Lakhs): "))
driven_kms = int(input("Enter Kilometers Driven: "))

fuel = input("Enter Fuel Type (Petrol/Diesel/CNG): ").strip().title()
selling_type = input("Enter Selling Type (Dealer/Individual): ").strip().title()
transmission = input("Enter Transmission (Manual/Automatic): ").strip().title()

owner = int(input("Enter Number of Previous Owners: "))
manufacture_year = int(input("Enter Manufacturing Year: "))
future_year = int(input("Enter Future Year (e.g. 2028): "))

future_car_age = future_year - manufacture_year

if future_car_age < 0:
    print("Future year cannot be less than manufacturing year.")
else:

    new_car = pd.DataFrame({
        "Car_Name": [le_car.transform([car_name])[0]],
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],        # <-- Correct column name
        "Fuel_Type": [le_fuel.transform([fuel])[0]],
        "Selling_type": [le_sell.transform([selling_type])[0]],
        "Transmission": [le_trans.transform([transmission])[0]],
        "Owner": [owner],
        "Car_Age": [future_car_age]
    })

    # Ensure feature order matches training data
    new_car = new_car[X.columns]

    predicted_price = model.predict(new_car)

    print("\n===================================")
    print(f"Estimated Selling Price in {future_year}: ₹ {predicted_price[0]:.2f} Lakhs")
    print("===================================")