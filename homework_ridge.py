import numpy as np
import math
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error,mean_absolute_error

# import data
data = pd.read_table(r'D:\360MoveData\Users\HP\Desktop\prostate.data.txt')
factors = [
 'lcavol',
 'lweight',
 'age',
 'lbph',
 'svi',
 'lcp',
 'gleason',
 'pgg45'
 ]
X = data [factors]
Y = data['lpsa']

def Standard_error(sample):
    std = np.std(sample,ddof=0)
    standard_error = std/math.sqrt(len(sample))
    return standard_error

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=1)#,test_size=0.2, random_state=1)

# set the final alpha by using RidgeCV
Lambdas = np.logspace(-5,2,200)
ridge_cv = RidgeCV(alphas=Lambdas,normalize=True,scoring='neg_mean_squared_error',cv=10)
ridge_cv.fit(X_train,y_train)
print ('Alpha is:'+str(round(ridge_cv.alpha_,4)))
ridge = Ridge(alpha=ridge_cv.alpha_)

# predict
ridge.fit(X_train, y_train)
y_predict=ridge.predict(X)
y_test_predict=ridge.predict(X_test)

# model evaluation (MSE,MAE,std_error)
mse_predict = round(mean_squared_error(y_test,y_test_predict),4)
mae_predict = round(mean_absolute_error(y_test,y_test_predict),4)
std_error = round(Standard_error(y_test_predict),4)

coef = []
for i in range(8):    
    coef.append((factors[i],round(ridge.coef_[i],4)))

print ('Intercept is:'+str(round(ridge.intercept_,4)))
print ('Estimated coefficients are:'+str(coef))
print ('Std Error is:'+str(std_error))
print ('MSE is:'+str(mse_predict))
print ('MAE is:'+str(mae_predict))
