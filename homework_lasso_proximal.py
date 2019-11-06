import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=1)

def Standard_error(sample):
    std = np.std(sample,ddof=0)
    standard_error = std/math.sqrt(len(sample))
    return standard_error

# Objective function: f(x) + lambda*norm1(x)
def obj(A,x,y,alpha):
    return f(A,x,y) + alpha*np.sum(np.abs(x))

# f(x) = (1/2)||Ax-y||^2
def f(A,x,y):
    Ax_y = A.dot(x) - y
    return 0.5*(Ax_y.T.dot(Ax_y))

# gradient of f(x)= A'(Ax - y)   
def grf(A,x,y):
    return A.T.dot(A.dot(x) - y)
    
# Model function evaluated at x and touches f(x) in xk
def m(x,xk,A,y,GammaK):
    innerProd = grf(A,xk,y).T.dot(x - xk)
    xDiff = x - xk
    return f(A,xk,y) + innerProd + (1.0/(2.0*GammaK))*xDiff.T.dot(xDiff)

# Shrinkage or Proximal operation
def shrinkage(x,kappa):
    return np.maximum(0.,x-kappa)-np.maximum(0.,-x-kappa)

def lasso_proximal(X,y,alpha=1,MAX_ITER=2000,beta=0.99,Gammak = 0.05):
    # beta is the decreasing factor for line search
    # Define parameters. Size of X is n x p
    X=np.array(X)
    y=[[i] for i in y]
   
    xk = [[0.48],[0.50],[-0.01],[0.11],[0.81],[-0.03],[0.13],[0.003]] # Initialize, other value may need more iteraions
    # Proximal Gradient Descent
    for k in range(MAX_ITER):
        # Line search
        while True:
            x_kplus1 = xk - Gammak*grf(X,xk,y)        # Gradient Descent (GD) Step
            if f(X,x_kplus1,y) <= m(x_kplus1,xk,X,y,Gammak):
                break
            else:
                Gammak = beta*Gammak
        x_kplus1 = shrinkage(x_kplus1,Gammak*alpha)   # Proximal Operation (Shrinkage)
        
        # Terminating Condition        
        Dobj = np.linalg.norm(obj(X,x_kplus1,y,alpha) - obj(X,xk,y,alpha))
        if(Dobj<0.01):
            break

        # Update xk
        xk = x_kplus1 
    return xk
        
#lasso_proximal(X,Y,alpha=0.0058)

coefficients = lasso_proximal(X_train,y_train,alpha=0.0058)
y_test_predict=X_test.dot(coefficients)

# model evaluation (MSE,MAE,std_error)
mse_predict = round(mean_squared_error(y_test,y_test_predict),4)
mae_predict = round(mean_absolute_error(y_test,y_test_predict),4)
std_error = round(Standard_error(y_test_predict),4)

coef = []
for i in range(8):
    coef.append((factors[i],round(coefficients[i][0],4)))

print ('Estimated coefficients are:'+str(coef))
print ('Std Error is:'+str(std_error))
print ('MSE is:'+str(mse_predict))
print ('MAE is:'+str(mae_predict))