# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:05:14 2020

@author: jai yadav
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


columns=['seismic','seismoacoustic','shift','genergy','gpuls','gdenergy','gdpuls','ghazard','energy','maxenergy','class']
df=pd.read_csv('C:/Users/jai yadav/Desktop/3RD SEMESTER/IC-272 DATA SCIENCE 3/assignment 4/seismic_bumps1.csv',usecols=columns)




Best_classification=[]     #define list which store the value of max value accuracry
def KNN(X_train,X_label_train,X_test):   #funtion to compute KNN
    best_acc=0                              #initilizing variable =0 for every new task
    print('Confusion Matrix:\tAccuracy score:\t\tk:\n')
    for n in [1,3,5]:                        #running fro loop over values for k = 1,2,3
        knn=KNeighborsClassifier(n_neighbors=n)        #computing knn for k =n
        knn.fit(X_train,X_label_train)       # Train the model using the training sets
        Prediction=knn.predict(X_test)
        
        
        print(metrics.confusion_matrix(X_label_test,Prediction),end='\t\t')    # computing confusion matrix for particular k
        print(round(metrics.accuracy_score(X_label_test,Prediction),3),end='\t\t\t')      #computing accuracy for the above computed confusion matrix
        print(n,'\n')
        if metrics.accuracy_score(X_label_test,Prediction) > best_acc:                # if condition to get  max accuracy for particular k
            best_acc=round(metrics.accuracy_score(X_label_test,Prediction),3)          #stroing max accuracy value for k = 1,2,3
            best_k=n
    Best_classification.append(best_acc)
    print('At k =',best_k,'accuracy is high at:',Best_classification,'\n')                   #give the value of k for which accuracy is max
    
    
#1-----------------------------------------------------------------
print('Solution1: ','\n')                
[X_train, X_test, X_label_train, X_label_test] =train_test_split(df[df.columns[:-1]],         ##used the command train_test_split from scikit-learn given below to split the data
                            df['class'], test_size=0.3, random_state=42,shuffle=True) 

pd.concat((X_train,X_label_train),axis=1).to_csv('seismic_bumps_train.csv',index=False)       #Saving the train data as seismic-bumps-train.csv
pd.concat((X_test,X_label_test),axis=1).to_csv('seismic_bumps_test.csv',index=False)          ##Saving the test data as seismic-bumps-test.csv

KNN(X_train,X_label_train,X_test)    #calling function knn to compute KNN

#2--------------------------------------------------------------------
print('Solution2: ','\n')
X_test=(X_test-X_train.min())/(X_train.max()-X_train.min())          #applying min-max  normalisation in the range 0 to 1 to the test data 
X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())        #applying min-max  normalisation in the range 0 to 1 to the train data

pd.concat((X_train,X_label_train),axis=1).to_csv('seismic_bumps_train_normalized.csv')   #Saving the normalized train data as seismic-bumps-train_normalized.csv
pd.concat((X_test,X_label_test),axis=1).to_csv('seismic_bumps_test_normalized.csv')     #Saving the normalized test data as seismic-bumps-test_normalized.csv

KNN(X_train,X_label_train,X_test)    #calling function Knn to compute knn for normalized data

#3----------------------------------------------------------------------
print('Solution3: ','\n')
X_train=pd.read_csv('seismic_bumps_train.csv')           #reading csv file containg train data
X_test=pd.read_csv('seismic_bumps_test.csv')       ##reading csv file containg test data

C0=X_train[X_train['class']==0][X_train.columns[0:-1]]   #train data having class = 0
Mean_C0=C0.mean().values                            #computing mean of the train data having class 0
Cov_C0=C0.cov().values                                  ##computing covariance of the train data having class 0
                     

C1=X_train[X_train['class']==1][X_train.columns[0:-1]]    #train data having class = 1
Mean_C1=C1.mean().values                            #computing mean of the train data having class 1
Cov_C1=C1.cov().values                               ##computing covariance of the train data having class 1

P_C0=len(C0)/(len(C0)+len(C1))        #compuiting prior of class 0
P_C1=len(C1)/(len(C0)+len(C1))       #compuiting prior of class 1
d=len(X_test.columns)-1

Predicted_class=[]
for x in X_test[X_test.columns[0:-1]].values:
    
     #compuiting likelihood of class 0
    p_x_C0=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_C0)**0.5)*np.e**(-0.5*np.dot(np.dot((x-Mean_C0).T,np.linalg.inv(Cov_C0)),(x-Mean_C0)))  
    #compuiting likelihood of class 1
    p_x_C1=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_C1)**0.5)*np.e**(-0.5*np.dot(np.dot((x-Mean_C1).T,np.linalg.inv(Cov_C1)),(x-Mean_C1)))   
    P_x=p_x_C0*P_C0+p_x_C1*P_C1   
    
    P_C0_x=p_x_C0*P_C0/P_x   #computing posterior probability for class 0
    P_C1_x=p_x_C1*P_C1/P_x    #computing posterior probability for class 1
    
    if P_C0_x>P_C1_x:Predicted_class.append(0)
    else:Predicted_class.append(1)
    
print('Confusion Matrix :\tAccuracy score :')
print(metrics.confusion_matrix(X_test[X_test.columns[-1]],Predicted_class),end='\t\t')    #printing confusion matrix
print(round(metrics.accuracy_score(X_test[X_test.columns[-1]],Predicted_class),3),'\n')    #printing accuracy for the above confusion matrix
Best_classification.append(round(metrics.accuracy_score(X_test[X_test.columns[-1]],Predicted_class),3))

#4--------------------------------------------------------------------
print('Solution4: ','\n')
Best_result=pd.DataFrame(Best_classification,index=['KNN','KNN on normalised data',
 'Bayes'],columns=['Accuracy'])
print(Best_result)
