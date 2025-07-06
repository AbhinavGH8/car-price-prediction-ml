
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Reading data
file = pd.read_csv("car_data_full.csv")
df = pd.DataFrame(file)

# Splitting dataset into input  and output to put in train_test_split
inputs = df.drop("price", axis=1)
outputs = df["price"]

# Separating numerical and categorical columns
nums = ["year", "mileage", "engine_power"]
cats = ["brand", "fuel_type", "transmission"]

# Splitting data into training and testing sets (70% train, 30% test)
xtrain, xtest, ytrain, ytest = train_test_split(inputs, outputs, test_size=0.3, random_state=42)

# Initializing preprocessing components
scaling = StandardScaler()
encoding = OneHotEncoder(handle_unknown="ignore")

# ColumnTransformer to apply appropriate preprocessing to each type of column
preprocessing = ColumnTransformer(
    [
        ("num", scaling, nums),         # Scale numerical features
        ("encode", encoding, cats)      # One-hot encode categorical features
    ]
)

# ----------------------------- Linear Regression Pipeline -----------------------------
LinearRegressionModel = Pipeline(steps=[
    ("preprocess", preprocessing),                # Apply preprocessing
    ("LinearRegression", LinearRegression())      # Train linear regression model
])

# Fitting linear regression model
LinearRegressionModel.fit(xtrain, ytrain)

# Predicting on test set for Linear Regression
Linearprediction = LinearRegressionModel.predict(xtest)

# ----------------------------- Decision Tree Regression Pipeline -----------------------------
DecisionTreeRegressionModel = Pipeline(steps=[
    ("preprocess", preprocessing),                                  # Apply preprocessing
    ("DecisionTree", DecisionTreeRegressor(                         # Decision Tree Regressor with hyperparameters
        max_depth=6,
        min_samples_leaf=6,
        min_samples_split=8
    ))
])

# Fitting decision tree model
DecisionTreeRegressionModel.fit(xtrain, ytrain)

# Predicting on test set for DecisionTree Regression
DecisionPrediction = DecisionTreeRegressionModel.predict(xtest)

# ----------------------------- Random Forest Regression Pipeline -----------------------------
RandomForestRegressionModel = Pipeline(steps=[
    ("preprocess", preprocessing),                                   # Apply preprocessing
    ("DecisionTree", RandomForestRegressor(                         # 100 estimators, fixed random state
        n_estimators=100,
        random_state=42
    ))
])

# Fitting random forest model
RandomForestRegressionModel.fit(xtrain, ytrain)

# Predicting on test set for Random Forestt
RandomForestPrediction = RandomForestRegressionModel.predict(xtest)

# ----------------------------- Comparing Predictions -----------------------------
for real, linearpred, decpred, randompred in zip(ytest[:10], Linearprediction[:10], 
                                                 DecisionPrediction[:10], RandomForestPrediction[:10]):
    print(f"Actual: {real}, Linear Prediction: {linearpred}\n"
          f"Decision Prediction: {decpred} Random Prediction: {randompred}\n")

# ----------------------------- Evaluation Metrics -----------------------------
# Mean Squared Error
linearmse = mean_squared_error(ytest, Linearprediction)
decimse = mean_squared_error(ytest, DecisionPrediction)
randmse = mean_squared_error(ytest, RandomForestPrediction)

# R² Score
linearr2 = r2_score(ytest, Linearprediction)
decir2 = r2_score(ytest, DecisionPrediction)
randr2 = r2_score(ytest, RandomForestPrediction)

# ----------------------------- Final Output -----------------------------
print(f"Linear Regression Performance: \nMSE: {linearmse}\nR2 Value: {linearr2}")
print() #for space 
print(f"Decision Tree Regression Performance: \nMSE: {decimse}\nR2 Value: {decir2}")
print()
print(f"Random Forest Regression Performance: \nMSE: {randmse}\nR2 Value: {randr2}")

#Now we implement sns and matplotlib to check how each regression compares

fig,axs = plt.subplots(1,2,figsize=(12,10))
pcolors=["#18c1db","#6be310","#438a0c"]
plt.title("Comparison Of Accuracy Of Regression Models")
axs[0].set_title("MSE Comparison")
axs[0].set_xlabel("Regression Models")
axs[0].set_ylabel("Error")
sns.barplot(x=["Linear","Decision Tree","Random Forest"]
            ,y=[linearmse,decimse,randmse],ax=axs[0],palette=pcolors)

axs[1].set_title("R2 Score Comparison")
axs[1].set_xlabel("Regression Models")
axs[1].set_ylabel("R2 Score")
sns.barplot(x=["Linear","Decision Tree","Random Forest"]
            ,y=[linearr2,decir2,randr2],ax=axs[1],palette=pcolors)
plt.show() #show the graph



