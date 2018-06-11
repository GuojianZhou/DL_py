#!/usr/bin/env python
#-*- encoding uft-8 -*-
# This code is based on the 
# https://blog.csdn.net/cymy001/article/details/79118741
# This is a DL python implement. 

X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]

from sklearn.preprocessing import PolynomialFeatures
poly2=PolynomialFeatures(degree=2)  #Ploy Special Value creator
x_train_poly2=poly2.fit_transform(X_train) #x_train_poly2 is the two poly special value from the training set

from sklearn.linear_model import LinearRegression
regressor_poly2=LinearRegression()
regressor_poly2.fit(x_train_poly2,y_train) #Model produce the two poly regression models

regressor=LinearRegression()   #Initialize the Linear Regression Model
regressor.fit(X_train,y_train)   #Create the model according to the training data set 

import numpy as np
xx=np.linspace(0,26,100)   #create the test data set
xx=xx.reshape(xx.shape[0],1)
yy=regressor.predict(xx)

xx_poly2=poly2.transform(xx) # two poly special value from the testing set
yy_poly2=regressor_poly2.predict(xx_poly2)

############Poly4################
poly4=PolynomialFeatures(degree=4)  #Ploy4 Special Value creator
x_train_poly4=poly4.fit_transform(X_train) #x_train_poly4 is the two poly special value from the training set

regressor_poly4=LinearRegression()
regressor_poly4.fit(x_train_poly4,y_train) #Model produce the poly4 regression models

xx_poly4=poly4.transform(xx) # poly4 special value from the testing set
yy_poly4=regressor_poly4.predict(xx_poly4)

############Poly4################

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label='Degree=2')
plt4,=plt.plot(xx,yy_poly4,label='Degree=4')
plt.axis([0,25,0,25])
plt.xlabel('independent variable')
plt.ylabel('dependent variable')
plt.legend(handles=[plt1,plt2,plt4])
plt.show()

print('The R-squared value of Polynomial Regressor(Degree=2) performing on the training data set is', regressor_poly2.score(x_train_poly2,y_train))

print('The R-squared value of Polynomial Regressor(Degree=4) performing on the training data set is', regressor_poly4.score(x_train_poly4,y_train))

X_test=[[6],[8],[11],[16]]
y_test=[[8],[12],[15],[18]]
print('Linear regression:',regressor.score(X_test,y_test))

X_test_poly2=poly2.transform(X_test)
print('Polynomial 2 regression:',regressor_poly2.score(X_test_poly2,y_test))

X_test_poly4=poly4.transform(X_test)
print('Polynomial 4 regression:',regressor_poly4.score(X_test_poly4,y_test))

from sklearn.linear_model import Lasso
lasso_poly4=Lasso() #Initialize the Lasso by default
lasso_poly4.fit(x_train_poly4,y_train) #Use the Lasso to regressor for the Poly4 model
print(' Lasso Poly4 ')
print(lasso_poly4.score(X_test_poly4,y_test)) # Value on the test data set
print(lasso_poly4.coef_) #Output Lasso model parameters list
print(' Regressor Poly4 ')
print(regressor_poly4.score(X_test_poly4,y_test))
print(regressor_poly4.coef_)

from sklearn.linear_model import Ridge
ridge_poly4=Ridge() # Initialize the Ridge by default
ridge_poly4.fit(x_train_poly4,y_train) #Use the Ridge to regressor the Poly4 model
print(' Ridge Ploy4 ')
print(ridge_poly4.score(X_test_poly4,y_test))
print(ridge_poly4.coef_) #Output Ridge model parameters list
print(np.sum(ridge_poly4.coef_**2))
print(' Regressor Ploy4 ')
print(regressor_poly4.coef_) #
print(np.sum(regressor_poly4.coef_**2))

