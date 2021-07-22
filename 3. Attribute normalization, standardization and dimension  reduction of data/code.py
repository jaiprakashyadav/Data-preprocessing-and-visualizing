# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:36:31 2020

@author: jai yadav
"""

import pandas as pd                           #importoing panadas
import numpy as np                            # importing numpy
import matplotlib.pyplot as plt               #importing matplotli.plt
from sklearn.decomposition import PCA         #importing pca 
data=pd.read_csv('C:/Users/jai yadav/Desktop/3RD SEMESTER/IC-272 DATA SCIENCE 3/assignment 3/landslide_data3.csv')   #reading the data file
data.drop(['dates','stationid'],axis=1,inplace=True)     #droping the attribute from given data frame 


#------------------------------------------------------------------------------
def solution1():    # defining funtion to Replace the outliers in any attribute with the median of the respective attributes
    n=0             #Initializing attribute count as 0
    for i in data.columns:    #running for loop across columns of data frame
        bottom_whis=2.5*np.percentile(data[i],25)-1.5*np.percentile(data[i],75) #computing bottom whisker Q1-1.5*IQR
        upper_whis=2.5*np.percentile(data[i],75)-1.5*np.percentile(data[i],25)  #computing upper whisker Q3+1.5*IQR
        outliers=data[i][(data[i]<bottom_whis) | (data[i]>upper_whis)] #Dataframe of outliers in attribute i
        data.iloc[outliers.index,n]=(data[i].drop(outliers.index)).median() #Replacing outliers with median of remaning values of df[i]
        n+=1 #Incrementing attribute count
        
print('Question 1:')
solution1()

#------------------------------------------------------------------------------

def solution1a():                                           #defining Function for computing Minimum and Maximum
    min_max=pd.concat((data_1a.min(),data_1a.max()),axis=1) #Dataframe with min and max of Df
    min_max=min_max.T                                       #Taking tranpose of data frame
    index=pd.Series(['Min','Max'])                          #defining index of data frame
    min_max.set_index([index],inplace=True)                 #Setting Index 
    print(min_max.T)                                        #printing dataframe with min and max of each attribute

print('a: ')    
print('\nBefore Min_Max Normalization:')
data_1a=data.copy()                                        #Copying data
solution1a()                                               #Function call

data_1a=(data-data.min())/(data.max()-data.min())*(9-3)+3  #Min_Max Normalization , setting min = 3 , max = 9
print('\nAfter Min_Max Normalization:')
solution1a()                                               #Function call

#------------------------------------------------------------------------------

def Solution1b():                                                       #Function for computing Mean and Standard Deviation 
    mean_std=pd.concat((round(data_1b.mean(),6),data_1b.std()),axis=1)  #Dataframe with Mean and Std Dev of Df
    mean_std=mean_std.T                                                 #Taking tranpose of data frame
    index=pd.Series(['Mean','Standard Dev'])                            #defining index of data frame
    mean_std.set_index([index],inplace=True)                            #Setting Index
    print(mean_std.T)                                                   #printing dataframe with min and max of each attribute

print('\nb)')
data_1b=data.copy()                                                     #Copying df
print('\nBefore Standardization:')
Solution1b()                                                            #Function call

data_1b=(data-data.mean())/(data.std())                                 #Standardize each attribute
print('\nAfter Standardization:')
Solution1b()                                                            #Function call

#------------------------------------------------------------------------------

def solution2():     #function for task 2
    
    print('a)')
    plt.scatter(D.T[0],D.T[1],marker='x',color='red')            #plotiing sactter plot of 1000 sample
    plt.xlabel('x1')                             #namimg ylabel as x1 
    plt.ylabel('x2')                             #naming ylabel as x2
    plt.title('Plot of 2D synthetic data')      #naming tiltle of plot as Plot of 2D synthetic data
    plt.show()
    
    print('\nb)')
    eigval,eigvec=np.linalg.eig(np.cov(D.T))                           #computing eignvalue and eigenvetore using inbuilt function
    print('Eigen values:',*eigval,'\nEigen vectors:',*eigvec.T)        #printing the eigen value and eigen veector
    plt.scatter(D.T[0],D.T[1],marker='x',color='red')                 #sactter plot of 1000 sample
    plt.quiver(0,0,eigvec[0],eigvec[1],angles="xy",color='black',scale=4) #plotting eigenvector 
    plt.axis("equal")
    plt.xlabel('x1')                                                   #namimg xlabel as x1
    plt.ylabel('x2')                                                   #namimg ylabel as x2 
    plt.title('Plot of 2D synthetic data and Eigen vectors')           #naming tiltle of plot as Plot of 2D synthetic data and Eigen Vector
    plt.show()
    
    
    
    print('\nc)')
    prj=np.dot(D,eigvec)                                               #Projection of data on eigenvector
    
    plt.scatter(D.T[0],D.T[1],marker='x',color='red')                  #sactter plot of 1000 sample
    plt.quiver(0,0,eigvec[0],eigvec[1],angles="xy",color='black',scale=3) #plotting eigenvector 
    plt.scatter(prj[:,0]*eigvec[0][0],prj[:,0]*eigvec[1][0],color='yellow',marker='x')  #plotting projection of datapoints on 1st eigenvector
    plt.axis("equal")
    plt.xlabel('x1')                                                   #namimg xlabel as x1
    plt.ylabel('x2')                                                   #namimg ylabel as x2
    plt.title('Projected values on 1st eigen vector')                  #naming title as Projected values on 1st eigen vector 
    plt.show()
    

    plt.scatter(D.T[0],D.T[1],marker='x',color='red')                   #sactter plot of 1000 sample
    plt.quiver(0,0,eigvec[0],eigvec[1],angles="xy",color='black',scale=3)  #plotting eigenvector
    plt.scatter(prj[:,1]*eigvec[0][1],prj[:,1]*eigvec[1][1],color='yellow',marker='x') #plotting projection of datapoints on 2nd eigenvector
    plt.axis("equal")
    plt.xlabel('x1')                                                   #namimg xlabel as x1
    plt.ylabel('x2')                                                   #namimg ylabel as x2
    plt.title('Projected values on 2nd eigen vector')                  #naming title as Projected values on 2nd eigen vector 
    plt.show()
    
    print('\nd)')
    D_=np.dot(prj,eigvec.T)                                        #Reconstructing Reduced Dimensional Data
    print('Root Mean Square Error:',(((D-D_)**2).sum()/len(D_)))   #computing RMSE between reduced data and sample data
    
print('\nQuestion 2:')
mean=np.array([0,0])                                         #mean matrix
cov=np.array([[5,10],[10,13]])                               #covriance matrix
D=np.random.multivariate_normal(mean,cov,1000,'ignore')      #Generating Data containg 1000 sample
solution2()                                                  #calling function

#------------------------------------------------------------------------------

def solution3():        #function for task 3
    print('\nQuestion 3:')
    data_frame=data_1b                    #copying standardized data from task 1b to data_frame

    #a
    print('a)')
    eigval,eigvec=np.linalg.eig(np.cov(data_frame.T))                #computing Eigen Value and Eigen Vector
    eigval.sort()                                                   #Sorting Eigen Values in Ascending Order
    eigval=eigval[::-1]                                            #Reversing (In Descending Order)

    pca=PCA(n_components=2)#PCA with l=2
    Data=pca.fit_transform(data_frame)
    

    for i in range(2):
        print('Variance along Eigen Vector',i+1,':',np.var(Data.T[i]),
              '\nEigen Value corresponding to Eigen Vector',i+1,':',eigval[i],'\n')

    plt.scatter(Data.T[0],Data.T[1],color='red')                            #Scatter plot of reduced dimensional data.
    plt.xlabel('Principal Component 1')                         #naming xlabel as Principal componenet 1
    plt.ylabel('Principal Component 2')                         #naming ylabel as Principal componenet 2
    plt.title('Scatter plot of reduced dimensional data')       #naming title of plot as Scatter plot of reduced dimensional data
    plt.show()

    #b
    print('\nb)')
    plt.bar(range(1,8),eigval,color='red')                                  #Bar graph of eigen values 
    plt.scatter(range(1,8),eigval,color='black')
    plt.plot(range(1,8),eigval,color='yellow')                                 #plots the graph
    plt.xlabel('Index')                                         #naming xlabel as Index
    plt.ylabel('Eigen Value')                                   #naming ylabel as Eigne value
    plt.title('Eigen Values in Descending Order')               #naming title as Eigen Values in Descending Order
    plt.show()


    #c
    print('c)')
    RMSE=[]                                                         #creating empty list
    for i in range(1,8):                                            #running for loop on the values of L
        pca=PCA(n_components=i)                                     #PCA with l=i
        Data=pca.fit_transform(data_frame)                          #Data with Reduced Dimension
        D_=pca.inverse_transform(Data)                              #Reconstructed Data
        RMSE.append((((data_frame.values-D_)**2).mean())**.5)              #Appending list with RMSE
    
    plt.bar(range(1,8),RMSE,color='red')          #Plotting Rmse error
    plt.plot(range(1,8),RMSE,color='yellow')          #plot the graph
    plt.scatter(range(1,8),RMSE,color='black')
    plt.ylabel('RMSE')                               #naming ylabel as RMSE
    plt.xlabel('L')                                  #naming xlabel as L
    plt.title('Reconstruction Error')                #naming title as Reconsturction error
    plt.show()

    
solution3()
