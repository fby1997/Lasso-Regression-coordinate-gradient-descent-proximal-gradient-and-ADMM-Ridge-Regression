import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
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

def lasso_admm(X,y,alpha=5,rho=1.,rel_par=1.,QUIET=True,\
                MAX_ITER=100,ABSTOL=1e-3,RELTOL= 1e-2):

    #Data preprocessing
    m,n = X.shape
    #save a matrix-vector multiply
    Xty = X.T.dot(y)

    #ADMM solver
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))

    # cache the (Cholesky) factorization
    L,U = factor(X,rho)

    # Saving state
    h = {}
    h['objval']     = np.zeros(MAX_ITER)
    h['r_norm']     = np.zeros(MAX_ITER)
    h['s_norm']     = np.zeros(MAX_ITER)
    h['eps_pri']    = np.zeros(MAX_ITER)
    h['eps_dual']   = np.zeros(MAX_ITER)

    for k in range(MAX_ITER):
        # x-update 
        tmp_variable = np.array(Xty)+rho*(z-u)[0] #(temporary value)
        if m>=n:
            x = spsolve(U,spsolve(L,tmp_variable))[...,np.newaxis]
        else:
            ULXq = spsolve(U,spsolve(L,X.dot(tmp_variable)))[...,np.newaxis]
            x = (tmp_variable*1./rho)-((X.T.dot(ULXq))*1./(rho**2))

        # z-update with relaxation
        zold = np.copy(z)
        x_hat = rel_par*x+(1.-rel_par)*zold
        z = shrinkage(x_hat+u,alpha*1./rho)

        # u-update
        u+=(x_hat-z)

        # diagnostics, reporting, termination checks
        h['objval'][k]   = objective(X,y,alpha,x,z)
        h['r_norm'][k]   = norm(x-z)
        h['s_norm'][k]   = norm(-rho*(z-zold))
        h['eps_pri'][k]  = np.sqrt(n)*ABSTOL+\
                            RELTOL*np.maximum(norm(x),norm(-z))
        h['eps_dual'][k] = np.sqrt(n)*ABSTOL+\
                            RELTOL*norm(rho*u)
        if (h['r_norm'][k]<h['eps_pri'][k]) and (h['s_norm'][k]<h['eps_dual'][k]):
            break
    return z.ravel(),h

def objective(X,y,alpha,x,z):
    return .5*np.square(X.dot(x)-y).sum().sum()+alpha*norm(z,1)

def shrinkage(x,kappa):
    return np.maximum(0.,x-kappa)-np.maximum(0.,-x-kappa)

def factor(X,rho):
    m,n = X.shape
    if m>=n:
       L = cholesky(X.T.dot(X)+rho*sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m)+1./rho*(X.dot(X.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U

coefficients=lasso_admm(X_train,y_train,alpha=0.0058)
y_test_predict=X_test.dot(coefficients[0])

# model evaluation (MSE,MAE,std_error)
mse_predict = round(mean_squared_error(y_test,y_test_predict),4)
mae_predict = round(mean_absolute_error(y_test,y_test_predict),4)
std_error = round(Standard_error(y_test_predict),4)

coef = []
for i in range(8):
    coef.append((factors[i],round(coefficients[0][i],4)))

print ('Estimated coefficients are:'+str(coef))
print ('Std Error is:'+str(std_error))
print ('MSE is:'+str(mse_predict))
print ('MAE is:'+str(mae_predict))
