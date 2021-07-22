# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:12:07 2020

@author: jai yadav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture  


#PART A----------------------------------------------------------------------

print('Solution of PART A')

#solution1
print('\n','Solution 1 (PART A) : ')
Train=pd.read_csv('C:/Users/jai yadav/Desktop/Assignment5/seismic_bumps_train.csv') # reading Train data
Test=pd.read_csv('C:/Users/jai yadav/Desktop/Assignment5/seismic_bumps_test.csv')  # reading Test Data
Train_0=Train[Train['class']==0]       #Train Data with Class 0
Train_1=Train[Train['class']==1]       #Train Data with Class 1

P_C0=len(Train_0)/len(Train)          #computing of Prior of Class 0
P_C1=len(Train_1)/len(Train)         #computing of Prior of Class 1

print('\nConfusion Matrix:\tAccuracy score:\t\tQ:\n')
best_acc=0               #Initializing list for computing Best Accuracy 
for Q in [2,4,8,16]:          #For loop for different values of Q
    GMM0 = sklearn.mixture.GaussianMixture(n_components=Q, covariance_type='full',random_state=20)  #GMM for Class 0
    GMM0.fit(Train_0[Train_0.columns[:-1]])                                         
    p_x_C0=np.exp(GMM0.score_samples(Test[Test.columns[:-1]]))     #computing Likelihood of x in Class 0
    
    GMM1 = sklearn.mixture.GaussianMixture(n_components=Q, covariance_type='full',random_state=20)  #GMM for Class 1
    GMM1.fit(Train_1[Train_1.columns[:-1]])
    p_x_C1=np.e**GMM1.score_samples(Test[Test.columns[:-1]])   #computing Likelihood of x in Class 1
    
    P_x=p_x_C0*P_C0+p_x_C1*P_C1      #computing Evidence
    np.seterr(invalid='ignore')    #for Ignoring the warning when P_x --> 0
    
    P_C0_x=p_x_C0*P_C0/P_x #computing Posterior Probability of Class 0
    P_C1_x=p_x_C1*P_C1/P_x #computing Posterior Probability of Class 1
    
    Predicted_class=(P_C1_x > P_C0_x)*1 #Array of Predicted Class
    
    print(sklearn.metrics.confusion_matrix(Test[Test.columns[-1]],Predicted_class),end='\t\t')          #Confusion Matrix for a particular Q
    print(round(sklearn.metrics.accuracy_score(Test[Test.columns[-1]],Predicted_class),5),end='\t\t\t') #Accuracy Score for a value of Q
    print(Q,'\n')
    
    if sklearn.metrics.accuracy_score(Test[Test.columns[-1]],Predicted_class) > best_acc:
            best_acc=round(sklearn.metrics.accuracy_score(Test[Test.columns[-1]],Predicted_class),5)    #Best Accuracy
            Q_best=Q          #Best value of Q
print('At Q =',Q_best,'accuracy is high at:',best_acc,'\n') #best accuracy 

#solution 2
print('\n','solution 2 (PART A) : ','\n')
Method=['KNN','KNN on normalized data','Bayes using unimodal Gaussian density','Bayes using GMM']
Accuracy=[0.92397,0.92397,0.88918,best_acc]                 #Accuracy from last assignment and best accuracy from above part
df=pd.DataFrame(Accuracy,index=Method,columns=['Accuracy']) #Dataframe of Methods and Accuracy
print(df,'\n')

#Part B-----------------------------------------------------------------------------------------------------
print('PART B')
df=pd.read_csv('C:/Users/jai yadav/Desktop/Assignment5/atmosphere_data.csv') #Dataframe
[X_train, X_test] =sklearn.model_selection.train_test_split(df, test_size=0.3, random_state=42,shuffle=True) #Splitting Data
X_train.to_csv('C:/Users/jai yadav/Desktop/Assignment5/atmosphere_train.csv') #Saving the train data to CSV file with Train Data
X_test.to_csv('C:/Users/jai yadav/Desktop/Assignment5/atmosphere_test.csv')   #Saving the test data to CSV file with Test Data

#Solution 1
print('\n','Solution 1 (PART B) :')

from sklearn.linear_model import  LinearRegression #importing function linear regression
regressor = LinearRegression()                  #assigning regressor = function linearregression
x=X_train['pressure'].values.reshape(-1,1)    #Reshaping the data
y=X_train['temperature'].values.reshape(-1,1) #Reshaping the data
regressor.fit(x,y)              #Fitting the data
y_pred = regressor.predict(x)   #Prediction on Train Data 

#a
print("\n(a) : ")
plt.scatter(x,y,alpha=0.75,label='Training Data',color='red')      #Scatter Plot of Training Data
plt.scatter(x,y_pred,marker='x',label='Best Fit Line',color='orange')  #Scatter Plot of Predicted Value on Training Data
plt.xlabel('Pressure')                                 #defining xlabel of the graph
plt.ylabel('Temperature')                              #defining ylabel of the graph
plt.title('Simple Linear Regression on Training Data')   #defining title of the graph
plt.legend();plt.show()

#b
print("\n(b) : ")
print('Root Mean Squared Error of Training Data :',end='')
print(((y-y_pred)**2).mean()**0.5)  #Computing RMSE of Training Data between predicted and true value

#c
print("\n(c) : ")
x=X_test['pressure'].values.reshape(-1,1)    #Reshaping the data
y=X_test['temperature'].values.reshape(-1,1) #Reshaping the data
y_pred=regressor.predict(x)         #Prediction on Test Data
print('Root Mean Squared Error of Test Data : ',end='')
print(((y-y_pred)**2).mean()**0.5)  #Computing RMSE of Test Data between predicted and true value

#d
print("\n(d) : ")
plt.scatter(y,y_pred,color='red') #Scatter Plot of Actual Temp vs Predicted Temp
plt.xlabel('Actual Temperature') #defining xlabel of the graph
plt.ylabel('Predicted Temperature') #defining ylabel of the graph
plt.title('Actual Temperature vs Predicted Temperature') #defining title of the graph
plt.xlim(5,35);plt.ylim(12.5,27.5)   #setting ranges for x and y coordinates
plt.show()

#Solution 2
print('\n','Solution 2 (PART B) :')

from sklearn.preprocessing import  PolynomialFeatures  #importing polynomialfeature 

min_RMSE_test=10**10 #setting the arbitary value to variable
RMSE_train=[];RMSE_test=[]  #Initializing to Empty List for storing RMSE error for different values of N

x_train=X_train['pressure'].values.reshape(-1,1)    #Reshaping the data
y_train=X_train['temperature'].values.reshape(-1,1)
x_test=X_test['pressure'].values.reshape(-1,1)
y_test=X_test['temperature'].values.reshape(-1,1)

N=[2,3,4,5] #running for loop over the order of polynomial 
for p in N: #p is degree of polynomial
    polynomial_features = PolynomialFeatures(degree=p)
    
    #a
    x_poly_train = polynomial_features.fit_transform(x_train)  #Transforming x_train in p dimension
    regressor = LinearRegression()
    regressor.fit(x_poly_train, y_train) #Fitting the data
    
    y_pred_train = regressor.predict(x_poly_train)             #Prediction on Training Data
    RMSE_train.append(((y_train-y_pred_train)**2).mean()**0.5) #Computing RMSE
   
    #b
    x_poly_test = polynomial_features.fit_transform(x_test)    #Transforming x_test in p dimension
    y_pred_test = regressor.predict(x_poly_test)               #Prediction on Training Data
    RMSE_test.append(((y_test-y_pred_test)**2).mean()**0.5)    #Computing RMSE
    
    if RMSE_test[-1]<=min_RMSE_test:
            y_best_pred_test=y_pred_test     #Best Prediction on Test Data
            y_best_pred_train=y_pred_train   #Best Prediction on Train Data     
        
plt.bar(N,RMSE_train,color='red') #Plot of RMSE of Train Data over the values of p
plt.scatter(N,RMSE_train,color='orange',s=100)
plt.plot(N,RMSE_train,color='black')
plt.xlabel('p');plt.ylabel('RMSE')
plt.xticks(N);plt.ylim(3.6,4.2)
plt.title('Training Data');plt.show()

plt.bar(N,RMSE_test,color='orange') #Plot of RMSE of Test Data  over the values of p
plt.scatter(N,RMSE_test,color='red',s=100)
plt.plot(N,RMSE_test,color='black')
plt.xlabel('p');plt.ylabel('RMSE')
plt.xticks(N);plt.ylim(3.6,4.2)
plt.title('Test Data');plt.show()

#c
plt.scatter(x_train,y_train,alpha=0.75,label='Training Data',color='red')                       #Scatter Plot of Train Data
plt.scatter(x_train,y_best_pred_train,marker='x',alpha=0.75,label='Best Fit Curve',color='orange') #Scatter Plot of Prediction on Train Data
plt.xlabel('Pressure');plt.ylabel('Temperature')  #defining Xlabel and ylabel
plt.title('Simple Nonlinear Regression on Training Data') #defining title
plt.legend();plt.show()

#d
plt.scatter(y_test,y_best_pred_test,color='red') #Scatter Plot of Actual Temp vs Predicted Temp
plt.xlabel('Actual Temperature');plt.ylabel('Predicted Temperature')   #defining Xlabel and ylabel
plt.title('Actual Temperature vs Predicted Temperature')  #defining title
plt.xlim(5,35);plt.ylim(12.5,27.5)
plt.show()
