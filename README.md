# Lasso-Regression-coordinate-gradient-descent-proximal-gradient-and-ADMM-Ridge-Regression
Use Ridge Regression and Lasso Regression in prostate cancer data

## Data Source
 http://web.stanford.edu/~hastie/ElemStatLearn/
## Environment Requirement
* Python3.6

    * sklearn 0.18.1
    * numpy 1.17.3
    * pandas 0.20.1

## Ridge Regression

This model contains regression and evaluation. You can also use your own data, just simply change the data path.

**Note**: The RidgeCV function is for finding Alpha, and I choose MSE, MAE and Standard Error to evaluate the model.

Result: 
* Alpha is:3.0721
* Intercept is:-0.2565
* Estimated coefficients are:[('lcavol', 0.4769), ('lweight', 0.4624), ('age', -0.0117), ('lbph', 0.1007), ('svi', 0.6278), ('lcp', 0.0045), ('gleason', 0.1349), ('pgg45', 0.0033)]
* Std Error is:0.1889
* MSE is:0.3931
* MAE is:0.4443

## Lasso Regression

This model is similar to Ridge, and I use coordinate gradient descent, proximal gradient and ADMM methods seperately to solve Lasso. 
