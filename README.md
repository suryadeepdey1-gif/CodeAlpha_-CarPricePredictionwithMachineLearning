1. **Import Libraries:** Imports `pandas` and `numpy` for data manipulation, `matplotlib` for generating charts, and essential modules from `sklearn` for data preprocessing, splitting, building the Random Forest Regressor, and calculating evaluation metrics.
2. **Load Dataset & Inspect:** Reads `car data.csv` into a Pandas DataFrame and prints the top 5 records along with column data types to inspect the raw structure.
3. **Check Missing Values:** Scans the dataset for missing or null entries and removes incomplete rows to ensure data quality.
4. **Feature Engineering:** Derives a new feature (`Car_Age`) by subtracting the manufacturing year from 2026, then removes the original `Year` column to improve predictive efficiency.
5. **Encode Categorical Data:** Uses `LabelEncoder` to transform text-based attributes (`Car_Name`, `Fuel_Type`, `Selling_type`, `Transmission`) into distinct numerical values that the machine learning algorithm can process.
6. **Prepare Features:** Separates input features (`X`) from the target variable (`y`), which is the car's `Selling_Price`.
7. **Split Dataset:** Uses `train_test_split` to allocate 80% of the dataset for training the model and 20% for evaluating performance (`random_state=42`).
8. **Train Model:** Instantiates a `RandomForestRegressor` with 100 decision trees and fits it on the training split (`X_train`, `y_train`).
9. **Prediction:** Evaluates test features (`X_test`) and creates a comparative side-by-side DataFrame showing actual prices versus predicted prices for the top 10 rows.
10. **Model Evaluation:** Calculates standard regression metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R-squared ($R^2$) score to measure prediction accuracy.
11. **Actual vs. Predicted Plot:** Displays a scatter plot mapping actual prices against model predictions alongside a red reference line indicating ideal performance.
12. **Feature Importance:** Plots a horizontal bar chart ranking input attributes based on how heavily the Random Forest relied on them to estimate prices.
13. **Predict Future Selling Price:** Provides an interactive terminal prompt where users can enter custom vehicle parameters (car model, showroom price, distance driven, fuel type, transmission, owners, manufacture year, and target future year) to calculate real-time estimated resale values.
