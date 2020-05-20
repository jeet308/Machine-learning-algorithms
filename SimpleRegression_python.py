#Simple Linear Regration algorith in python

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import style

style.use('fivethirtyeight')

# simple Linear Regression eqaution y = mX+b

X = np.array([1,5,10,15,20,25],dtype='float64')
y = np.array([55,65,75,73,87,81],dtype='float64')

def slope(x, y):
    m = (((mean(x) * mean(y)) - (mean(x * y))) /
         ((mean(x) ** 2) - (mean(x ** 2))))
    return m

def y_intercept(m, x, y):
    b = (mean(y) - (m * mean(x)))
    return b

def square_error(y_orig, y_line):
    return sum((y_line - y_orig) ** 2)

def Coefficent_of_determination(y_orig, y_line):
    #print(y_orig)
    #print(y_line)
    y_mean_line = [mean(y_orig) for y in y_orig]
    squre_error_reg = square_error(y_orig, y_line)
    squre_error_y_mean = square_error(y_orig, y_mean_line)
    return 1 - (squre_error_reg / squre_error_y_mean)

m=slope(X,y)
b=y_intercept(m,X,y)

print('b(intercept of y) : ',y_intercept(m,X,y))
print('m(slope) : ',slope(X,y))

regration_line = [(m*x)+b for x in X ]
r_squre = Coefficent_of_determination(y,regration_line)
print('R^2 score(coefficient of determination) : ',r_squre)

plt.plot(X,regration_line,color='blue')
plt.scatter(X,y,color='orange',s=100)
plt.xlabel('X-independent variable')
plt.ylabel('y-dependent variable')
plt.title('Simple Linear Regration')
plt.show()
