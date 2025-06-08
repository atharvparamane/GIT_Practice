# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate a simple dataset
# X represents the input features, y represents the target values
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.2, 4.1, 6.0, 8.1, 10.2])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Display the model's coefficients
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict a new value
new_value = np.array([[6]])
prediction = model.predict(new_value)
print("Prediction for input 6:", prediction[0])
