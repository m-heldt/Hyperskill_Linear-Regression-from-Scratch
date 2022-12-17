# write your code here
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CustomLinearRegression(object):

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = 0.0
        self.intercept = np.array(0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.fit_intercept == True:
            x_len = X.shape[0]
            ones = np.array([list(np.ones(x_len))])
            X = np.concatenate((ones, X.to_numpy().T)).T
        else:
            X = X.to_numpy()
        y = y.to_numpy()

        B = inv(X.T @ X) @ X.T @ y

        if self.fit_intercept == True:
            self.intercept = B[0]
            self.coefficient = np.array(B[1:])
        else:
            self.coefficient = np.array(B)

        return B

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y = X.to_numpy() @ self.coefficient + self.intercept
        return y

    def r2_score(self, y: pd.Series, yhat: np.ndarray) -> float:
        up = np.sum((y - yhat)**2)
        bottom = np.sum((y - y.mean())**2)

        return 1 - up/bottom

    def rmse(self, y, yhat) -> float:
        mse = np.sum((y - yhat)**2)/len(y)
        return np.sqrt(mse)

# df = pd.DataFrame({
#     'x': [4.,4.5,5.,5.5,6.,6.5,7.0],
#     'w': [1,-3,2,5,0,3,6],
#     'z': [11,15,12,9,18,13,16],
#     'y': [33,42,45,51,53,61,62]
#                    })

# df = pd.DataFrame({
#     'Capacity': [0.9,0.5,1.75,2.0,1.4,1.5,3.0,1.1,2.6,1.9],
#     'Age': [11,11,9,8,7,7,6,5,5,4],
#     'Cost/ton': [21.95,27.18,16.9,15.37,16.03,18.15,14.22,18.72,15.4,14.69]
#                    })

df = pd.read_csv('data_stage4.csv')

regCustom = CustomLinearRegression(fit_intercept=True)
regCustom.fit(df[['f1', 'f2', 'f3']], df['y'])
y_pred = regCustom.predict(df[['f1', 'f2', 'f3']])
# print(y_pred)

regSci = LinearRegression(fit_intercept=True)
regSci.fit(df[['f1', 'f2', 'f3']], df['y'])
y_pred_Sci = regSci.predict(df[['f1', 'f2', 'f3']])

output = {
    'Intercept': regCustom.intercept - regSci.intercept_,
    'Coefficient': regCustom.coefficient - regSci.coef_,
    'R2': regCustom.r2_score(df['y'], y_pred) - r2_score(df['y'], y_pred_Sci),
    'RMSE': regCustom.rmse(df['y'], y_pred) - np.sqrt(mean_squared_error(df['y'], y_pred_Sci))
}
print(output)
