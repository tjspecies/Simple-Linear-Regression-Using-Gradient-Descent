# Simple Linear Regression Using Gradient Descent

Overview: This project implements a simple linear regression model using the gradient descent algorithm. The goal is to analyze the relationship between two variables (x and y), compute slope and intercept, and evaluate model performance with Mean Squared Error (MSE) and R-squared (R²). The dataset myData.csv contains 100 data points.

# Dataset
File: myData.csv
Attributes: x (independent variable) and y (dependent variable)

# Methodology
Implemented gradient descent manually in Python (no libraries except Matplotlib for visualization).
Initial parameters:
Slope (m) = 0
Intercept (b) = 0
Learning rate = 0.000001
Iterations = 1000

At each iteration, slope and intercept were updated using gradient descent.
Predictions were computed as: y_pred = m * x + b.
Logged MSE at every iteration and evaluated model fit with R².
Generated regression line and MSE convergence plots.

# Implementation
Dataset read manually from myData.csv.
Gradient descent algorithm coded from scratch.

# Outputs:
SLRTraining[1000][1e-06]MSEs.txt – MSE log
SLRModelParameters.txt – Final slope, intercept, MSE, R²
RegressionPlot.png – Scatter plot with regression line
MSE_Per_Iteration_Plot.png – Error curve over iterations

# Results
Slope (m): 1.4689
Intercept (b): 0.0295
R²: 0.5890
The slope suggests each unit increase in x raises y by ~1.47.
The intercept is close to zero, meaning predictions align near the origin.
R² = 0.589 indicates a moderate fit (about 59% of variance explained).

# Discussion
Regression line shows a moderate positive relationship between x and y.
MSE decreased sharply early on, flattening between iterations 600–800, showing convergence.
Challenges included data preprocessing, gradient descent tuning, and numerical stability.
Strengths: straightforward gradient descent implementation, effective error minimization.

# Conclusion
This project demonstrates how gradient descent can be used for simple linear regression. Results show moderate predictive accuracy. Future work may include scaling features, hyperparameter tuning, or extending to multivariate regression.

# References
DATA 527 course materials
Python documentation
Online resources on linear regression and gradient descent
