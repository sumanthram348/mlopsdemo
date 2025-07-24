# Import Libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('data/rental_1000.csv')

# Define features (X) and labels (y)
X = df[['rooms', 'sqft']].values
y = df['price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Save the trained model to disk
filename = 'model/rental_prediction_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Make predictions using the testing data
y_pred = model.predict(X_test)

# Calculate the Root Mean Squared Error (RMSE) of the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: ", rmse)

print("Model Trained Successfully Comppleted")