import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:\\Users\\SHREEEE\\Desktop\\mini project\\insurance.csv")

# Display the first few rows of the dataset
print("Dataset Head:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualize the number of smokers by gender using Plotly
figure = px.histogram(data, x="sex", color="smoker", title="Number of Smokers by Gender")
figure.show()

# Encode categorical variables
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
print("\nEncoded Data Head:")
print(data.head())

# Visualize the distribution of regions using a pie chart
pie_data = data["region"].value_counts()
fig = px.pie(values=pie_data.values, names=pie_data.index, title="Distribution of Regions")
fig.show()

# Calculate and display the correlation matrix
print("\nCorrelation Matrix:")
print(data.corr())

# Define features and target variable
X = data[["age", "sex", "bmi", "smoker"]]
y = data["charges"]

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
forest = RandomForestRegressor(random_state=42)
forest.fit(xtrain, ytrain)

# Make predictions on the test set
ypred = forest.predict(xtest)

# Create a DataFrame for the predicted values
predictions = pd.DataFrame({"Predicted Premium Amount": ypred})
print("\nPredictions:")
print(predictions.head())

# Evaluate the model
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Optional: Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(ytest, ypred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel('Actual Premium Amount')
plt.ylabel('Predicted Premium Amount')
plt.title('Actual vs Predicted Premium Amount')
plt.show()
