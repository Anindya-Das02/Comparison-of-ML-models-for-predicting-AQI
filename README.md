# Comparison-of-ML-models-for-predicting-AQI

### Goal ###
In this project we are comparing various machine learning models to find which model works better for predicting the AQI (Air Quality Index).

### Machine learning models used ###
In this project we are using regression models such as:
* Multiple Linear Regression
* Polynomial Regression
* Decision Tree Regression
* Random Forest Regression
* Support Vector regression (SVR)

### Result ###
Models | R^2 | RMSE | MAE | RMSLE
-------|-----|------|-----|------
MLR    |0.9965| 5.4973 | 3.4796 | 0.0517
Decision Tree | 0.9955 | 6.2370 | 2.354 | 0.0563 
Random Forest |0.9982| 3.8577 | 1.7016 | 0.0422
SVR | 0.9164 | 27.0025 | 19.0722 | 0.1686
Poly R | -4.1417 | 211.8759 | 81.5855 | 0.4638

### Conclusion ###
From the above table it is evident that the _Random Forest Regressor_ performed the best out of all other regression models.
