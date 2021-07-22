# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:57:24 2020

@author: jai yadav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data1=pd.read_csv("C:/Users/jai yadav/Desktop/3RD SEMESTER/IC-272 DATA SCIENCE 3/assignment 2/pima_indians_diabetes_miss.csv")  #data frame with missing values
data2=pd.read_csv("C:/Users/jai yadav/Desktop/3RD SEMESTER/IC-272 DATA SCIENCE 3/assignment 2/pima_indians_diabetes_original.csv")  #data frame with original values
list1=['pregs','plas','pres','skin','test','BMI','pedi','Age','class']   #list with column names
list3=['mean','median','mode','standard-deviation']

def solution1(data1,list1):  # defining function to find the number of null values in data frame
    for i in list1:
        nullvalue= data1[i].isnull().sum()   # used the inbuilt function isnull() to find null values
        plt.bar(i,nullvalue)        # for creating bar plot
        plt.xlabel('Attributes')             #defining x label for the plot
        plt.ylabel('missing values')         #defining the y label for the plot
        plt.title('No. of missing values')   # defining the title of the plot
    plt.grid(True)                           #to show grid lines in the plot
    plt.show()
        

def solution2a(data1):       # defining function to drop the tuples having equal to or more than one third of attributes with missing values
    index_tuple=[]       #defining the list to store index of null tuple
    total_tuple=0            # variable to store total number of deleted tuble
    for i in range(len(data1.index)) :            # run the foor loop on every tuple
        nullvalue=data1.iloc[i].isnull().sum()    # storing number of missing value in tuple i
        if nullvalue >= 3 :                         ## checking the tuples (rows) having equal to or more than one third of attributes with missing values
            index_tuple.append(i)      # appending tuple index
            total_tuple+=1
    print('Total no. of tuples deleted:',total_tuple)
    print('\n')
    print('Row no. of deleted tuples:\n',*index_tuple,'\n')
    data1.drop(index_tuple,inplace=True)              #deleteing tuples from data frame
    


def solution2b(data1):              # defining function to Drop the tuples having missing value in the target (class) attribute.
    index_tuple_class=data1[data1['class'].isna()]   #checking for the null value in class attribute
    total_tuple=len(index_tuple_class)                # total number of tuple with missing value in class attribute
    print('Total no. of tuples deleted of class attribute:',total_tuple)
    print('\nRow no. of deleted tuples:\n',*index_tuple_class.index,'\n')
    data1.drop(index_tuple_class.index,inplace=True)     #droping the tuple having missing value in class attribute
    


def solution3(data1,list1):       #defining functions to count the number of missing values in each attributes
    print('Missing value in each attribute : ','\n')
    total=0            #variable for summing total missing value
    for i in list1:
        nullvalue= data1[i].isnull().sum()   #checking for null value in  i column
        total+=nullvalue       
        plt.bar(i,nullvalue)              # for creating bar plot between attribute and mssing value
        plt.xlabel('Attributes')           # x label for bar plot
        plt.ylabel('missing values')      # y label for bar plot
        plt.title('No. of missing values')    # total label for bar plot
        print(i+' :' + str(nullvalue))          
    
    print('\n')
    print("total number of missing value : " + str(total))      # giving total null value in file
    plt.grid(True)
    plt.show()
    
def solution4a1(data,data_old):       #defining function to Compute the mean, median, mode and standard deviation for each attributes and compare the same with that of the original file
    
    data=pd.concat((data.mean(),data.median(),data.mode().loc[0],        #data frame with mean , median , mode , standard deviation o f each column
            data.mode().loc[1],data.std()),axis=1,)
    index=[['Mean','Median','Mode1','Mode2','Statndard Dev']]    #defining index
    data=data.T
    data.set_index(index,inplace=True)   #setting index
    data=data.T
    print('New Data:\n')
    print(data)     
    
    data_old=pd.concat((data_old.mean(),data_old.median(),data_old.mode().loc[0],        #data frame  with mean , median , mode , standard deviation o f original data of each column
            data_old.mode().loc[1],data_old.std()),axis=1,)
    data_old=data_old.T
    data_old.set_index(index,inplace=True)   # setting index
    data_old=data_old.T
    print('\nOriginal Data:\n')
    print(data_old)


        
def solution4a2(data_new):  # defining function to Calculate RMSE between the original and replaced values for each attribute
    print('RMSE between the original and replaced values for each attribute :\n')
    for i in data1.columns:      # running for loop for each column 
            ln=0
            RMSE=0
            null_index=data1[i][data1[i].isna()].index        # finding the location of null places in data fram
            for j in null_index:                            #running for loop over index of null values in particular attribute
                RMSE+=(data_new[i][j]-data2[i][j])**2      # computing rmse between original data and missing place which is now filled with mean of attribute
                ln+=1
            if ln==0:
                RMSE=0.0
            else :
                RMSE/=ln
                RMSE**=0.5
            print(i,':',RMSE)      # giving rmse value for each attribute
            plt.bar(i,RMSE)     # plotting bar graph between rmse and attribute
            plt.xlabel('Attributes') # defining x label
            plt.ylabel('RMSE')    # defining y label
            plt.title('RMSE B/W the original and replaced value')   # defining title for the plot
    plt.grid(True)        # to show grid lines in the plot
    plt.show()
             

def outliers(x):  # defining function to calculate outlir for attrinute x
    minimum=2.5*np.percentile(data_LI[x],25)-1.5*np.percentile(data_LI[x],75)  # computing  Q1- 1.5*IQR
    maximum=2.5*np.percentile(data_LI[x],75)-1.5*np.percentile(data_LI[x],25)   #  computing Q3 +1.5*IQR
    outliers_=pd.concat((data_LI[x][data_LI[x]> maximum],data_LI[x][data_LI[x]< minimum]))       # data frame with outliers 
    return outliers_  # returning the data frame


def solution5a(data):      # defining unction to draw boxplot
    plt.figure(figsize=(10,2))                 #define the size of boxplot
    data.boxplot(column=['Age'],notch='True',grid=True,vert=False,patch_artist=False,widths=0.1)      #boxplot for age attribute
    plt.title('Boxplot')     # defining title for boxplot
    plt.xlabel('Age in year')    # defining x label for boxplot
    plt.show()
    
    plt.figure(figsize=(10,2))                   #define the size of boxplot
    data.boxplot(column=['BMI'],notch='True',grid=True,vert=False,patch_artist=False,widths=0.1)    #boxplot for BMI attribute
    plt.title('Boxplot')    # defining title for boxplot
    plt.xlabel('BMI')       # defining x label for boxplot
    plt.show()
    

print('\nSolution to Task 1 : ','\n')
solution1(data1,list1)       # calling function solution1


print('\nSolution to Task 2(a) : ', '\n')
solution2a(data1)              # calling function solution2a
print('\nSolution to Task 2(b) : ', '\n')
solution2b(data1)              # calling function solution2b



print('\nSolution to Task 3 : ', '\n')
solution3(data1,list1)            # calling function solution3a



print('\nSolution to Task 4(a)-i : ', '\n')
print('\nMissing values replaced by mean of their respective attribute','\n')
data_mean=data1.fillna(data1.mean())       # replacing the null values with mean values of the attribute 
solution4a1(data_mean,data2)                  #calling  function solution4a1     solution for question 4 part 1
print('\nSolution to Task 4(a)-ii : ', '\n')
solution4a2(data_mean)            #calling  function solution4a2


data_LI = data1.interpolate()            # replacing the null value by interpolation method
print('\nSolution to Task 4(b)-i : ', '\n')
print('\nMissing values in each attribute replaced using linear interpolation technique','\n')
solution4a1(data_LI,data2)             #calling  function solution4a1    solution for question 4 part 2
print('\nSolution to Task 4(b)-ii : ', '\n')
solution4a2(data_LI)                   #calling  function solution4a2    solution for question 4 part 2 


print('\nSolution to Task 5-i : ', '\n')
print('\noutliers in the attributes Age : ','\n')
print(*outliers('Age').values,'\n')               # outliers in age boxplot
print('\noutliers in the attributes BMI : ','\n')
print(*outliers('BMI').values,'\n')               # outliers in BMI boxplot

print('\nSolution to Task 5-i : ', '\n')
solution5a(data_LI)
print('\nSolution to Task 5-ii : ', '\n')
print('\nBoxplot (outliers replaced by median of the attribute)','\n')
outliers_age=outliers('Age')                    # outliers after outliers replaced by median
data_LI['Age'][outliers_age.index]=data_LI['Age'].median()        
outliers_bmi=outliers('BMI')
data_LI['BMI'][outliers_bmi.index]=data_LI['BMI'].median()      #replacing outlier with median


solution5a(data_LI)        #draw  boxplot    
