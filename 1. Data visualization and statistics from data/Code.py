#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:33:05 2020

@author: Jai Prakash Yadav
Roll.No =B19247
mobile no- 9306871378

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
list1=['temperature','humidity','pressure','rain','lightmax','moisture','lightavgw/o0']
list2=['mean','mode','median','minimum','maximum','standard-deviation']
data=pd.read_csv("C:/Users/jai yadav/Desktop/landslide_data3.csv")

def solution1():            #function solution1() to find mean, median ,mode, max value , min value , standard deviation for each field.
    for i in list2:
                                     #running nested loop
        for j in list1:            
            if(i=='mean'):
                print(i,j,'=',data[j].mean())    #computing mean for each of the field
            elif(i=='mode'):
                print(i,j,'=',(data[j].mode())[0])    #computing mode for each of the field
            elif(i=='median') :
                print(i,j, '=',data[j].median())       #computing median for each of the field
            elif(i=='maximum'):
                print(i,j,'=',data[j].max())          #computing maximum value for each of the field
            elif(i=='minimum'):
                print(i,j,'=',data[j].min())          #computing minimum value for each of the field
            elif(i=='standard-deviation'):
                print(i,j,'=',data[j].std())         #compuing standard deviation for each of the field
                
        print()
        


def solution2and3(data,list1):          #function solution2and3 to compute scatter plot and compute correlation between fields
    list3=['rain','temperature']
    
    for i in list3:                            #running nested loop
        for j in list1:
            if (j!=i):
                data.plot.scatter(color='red',x=i,y=j)             #plotting scatter plot between different fields   
                plt.grid(True)
                plt.title('Scatter Plot between '+ i + ' and '+ j)
                print('Correlation between',i,'and',j,'=',data[i].corr(data[j]))     #computing correlation between different fields according to question
                  
    plt.show()        
    

            
def solution4(data):  #function solution4 to plot histogram of rain and moisture          
    list4=['rain','moisture']
                     
    for i in list4:               #running loop to plot histogram
        
        plt.figure(figsize=(10,7)) 
        plt.title('Histogram Plot of '+i)
        plt.xlabel(i)
        data[i].plot.hist(color='magenta',bins=30)     #plotting histogram 
        plt.grid(True)
        plt.show()
        
             
           
    
def solution5(data):    #function solution5 to plot histogram of rain for each station id
                              
    y=data.groupby('stationid')    #Using “groupby” function to group the sensors according to their “stationid”

    fig,axs=plt.subplots(2,5,figsize=(22,10))
    r=0;c=0
    for i in y['rain']:
        axs[r][c].grid(True)
        axs[r][c].hist(i[1])
        axs[r][c].set_title('Sensor:'+i[0])   #title = sensor: ti
        axs[r][c].set_xlabel('Rain in ml')    #xlabel =  Rain in ml
        axs[r][c].set_ylabel('Frequency')     #ylabel = frequeency
        c+=1
        if c==5:
            r=1;c=0
    plt.show()
    
    
def solution6(data):      
    plt.figure(figsize=(10,2))                 #function solution6 for obtaining the boxplot for the attributes ‘rain’ and ‘moisture'.
    data.boxplot(column=['rain'],notch='True',grid=True,vert=False,patch_artist=False,widths=0.1)      #boxplot for rain
    plt.title('Rain')
    plt.xlabel('Rain in ml')
    plt.show()
    
    plt.figure(figsize=(7,7))
    data.boxplot(column=['moisture'],notch='True',grid=True,vert=True,patch_artist=False,widths=0.1)    #boxplot for moisture
    plt.title('Moisture')
    plt.ylabel('Moisture in percentage')
    plt.show()


print("Solution for Question 1 :\n")
solution1()
print("Solution for Question 2 and 3 :\n" )
solution2and3(data,list1)
print("Solution for Question 4\n")
solution4(data)
print("Solution for Question 5\n")
solution5(data)
print("Solution for Question 6\n")
solution6(data)

