import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle


df = pd.read_csv('calories.csv')  


X = df[['Minutes']]
y = df['CaloriesBurned']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr = LinearRegression()
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


minutes = [[50]]  
pred_calories = lr.predict(minutes)
print(f"Predicted Calories Burned for 50 minutes: {pred_calories[0]:.2f}")


with open('model.pkl', 'wb') as f:
    pickle.dump(lr, f)

print("Model saved as model.pkl")
