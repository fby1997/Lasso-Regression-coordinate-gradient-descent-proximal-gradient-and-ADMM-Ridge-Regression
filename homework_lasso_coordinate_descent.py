import numpy as np
import math
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso,LassoCV
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

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=1)

# set the final alpha by using LassoCV
Lambdas = np.logspace(-5,5,200)
lasso_cv = LassoCV(alphas=Lambdas,normalize=True,cv=10)
lasso_cv.fit(X_train,y_train)
print ('Alpha is:'+str(round(lasso_cv.alpha_,4)))
lasso = Lasso(alpha=lasso_cv.alpha_)

# predict
lasso.fit(X_train,y_train)
y_predict = lasso.predict(X)
y_test_predict = lasso.predict(X_test)

# model evaluation (MSE,MAE,std_error)
mse_predict = round(mean_squared_error(y_test,y_test_predict),4)
mae_predict = round(mean_absolute_error(y_test,y_test_predict),4)
std_error = round(Standard_error(y_test_predict),4)

coef = []
for i in range(8):
    coef.append((factors[i],round(lasso.coef_[i],4)))

print ('Intercept is:'+str(round(lasso.intercept_,4)))
print ('Estimated coefficients are:'+str(coef))
print ('Std Error is:'+str(std_error))
print ('MSE is:'+str(mse_predict))
print ('MAE is:'+str(mae_predict))
