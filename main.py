import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/Sai kumar/Downloads/archive (4)/BostonHousing.csv")

 #Preprocess Data
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

 #Fill missing values in 'rm' with mean
df['rm'].fillna(df['rm'].mean(), inplace=True)

#Feature Selection
# Choosing important features
X = df[['rm', 'lstat', 'ptratio', 'indus', 'tax']]
y = df['medv']

#Model Building
model = LinearRegression()

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Model
model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)

# Calculate metrics
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

#  Visualize Predictions vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()
