# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 13:29:47 2020

@author: jai yadav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df=pd.read_csv('C:/Users/jai yadav/Desktop/assignment 6/datasetA6_HP.csv')



#------------------------------------------------------------------------------------------------------------

#QUESTION 1
print('\n','Solution 1:')


#solution to question 1 part a
print('\na)')
df['Date']=pd.to_datetime(df['Date'],dayfirst=True)
plt.plot(df['Date'],df['HP'],color='red')     #plot of date and power consumed 
plt.xlabel('Date')                           #defining xlabel of the plot as date
plt.ylabel('Power Consumed (MW)')             #defining ylabel of the plot as power consumed
plt.xticks(rotation='45')                    #rotating xticks by 45 
plt.title('Power consumed (in MW) vs. days')
plt.show()



#solution to question 1 part b
print('\nb)\n')
x_t=df['HP']    #defining data frame 
x_t_1=df['HP'][1:]  #defining dta frame for 1 day lagged sequence
print('Correlation coefficient between x_t and x_t-1 : ',end='')
print(np.corrcoef(x_t_1,x_t[:-1])[0][1])      #computing corelation between one day lag time and given time sequence



#solution to question 1 part c
print('\n\nc)')
plt.scatter(x_t[:-1],x_t_1,color='red')     #plot between given time sequence and one day lagged time sequence
plt.xlabel('Given Time Sequence')           #defining xlbel of the plot as given time sequence
plt.ylabel('One-Day Lagged Generated Sequence')    #defining ylabel of the plot as one day lagged generated sequence
plt.title('Scatter plot one day lagged sequence vs. given time sequence')
plt.show()



#solution to question 1 part d
print('\nd)')
x_t=df['HP']            #given data frame
Correlation=[]           #list containg correlation for different value of p
for p in range(1,8):       #running for loop over value of p
    x_t_p=df['HP'][p:]        #lagging the data by p days
    Correlation.append(np.corrcoef(x_t_p,x_t[:-p])[0][1])   #computing correlation between the given sequence and p day lagged time sequence
plt.plot(range(1,8),Correlation,color='red')           # plot between coorelation and corresponding laged p value
plt.scatter(range(1,8),Correlation,marker='o',color='black')
plt.ylabel('Correlation Coefficient')   #defining ylabel as correlation coefficient 
plt.xlabel('Lagged Value')             #defining xlabel as lagged value
plt.title('Autocorrelation')          #defining title as autocorrelation
plt.show()



#solution to question 1 part e
print('\ne)')
sm.graphics.tsa.plot_acf(df['HP'],lags=range(1,8))
plt.ylabel('Correlation Coefficient');plt.xlabel('Lagged Value')
plt.show()


#-----------------------------------------------------------------------------------------------------------

#solution to question 2
print('\n','Solution 2:')
train, test = x_t[:len(x_t)-250], x_t[len(x_t)-250:]
test_t=test.values[1:]  #slice  1 data value from the starting
test_t_1=test.values[:-1]  #slice 1 data value from last
print('\nRMSE of Persistance Model : ',end='')
print(((test_t-test_t_1)**2).mean()**0.5)   #computing rmse for persistence model


#---------------------------------------------------------------------------------------------------------------------------


#QUESTION 3
print('\n','Solution 3:',)



#Solution to question 3 part a
print('\na)')
    
# train auto regression    
model=AR(train)
model_fit = model.fit(maxlag=5)

#make predictions
prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

rmse=((prediction-test)**2).mean()**0.5    #computing rmse btween predicted value and original value
print('\nRMSE of AR(5) Model : ',end='')
print(rmse)

plt.scatter(test,prediction,color='red')  #plot between predicted test data nad original test data
plt.xlabel('Original Test Data')    #defining xlabel of plot as original dest data
plt.ylabel('Predicted Test Data')   #defining ylabel of the plot as predicted test data
plt.title('AR(5)')                  #setting title as AR(5)
plt.show()   



#Solution to question 3 part b
print('\nb)\n')
print('p:\t RMSE:\n')
Lag=[1,5,10,15,25]
RMSE=[]    #creating list to store rmse values for different time lag
for p in Lag:    #running for loop for different time lag
    
    # train auto regression   
    model=AR(train)
    model_fit = model.fit(maxlag=p)
    
    #make predictions
    prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    
    rmse=((prediction-test)**2).mean()**0.5    #compute rmse value between predicted value and original vale
    RMSE.append(rmse)          #append rmse value in list created above
    print(p,'\t',rmse)



#Solution to question 3 part c
print('\nc)\n')
x_t=train

for p in range(1,len(train)):   #running for loop for train data
    x_t_p=train[p:]
    if abs(np.corrcoef(x_t_p,x_t[:-p])[0][1]) < 2/len(train)**0.5:        #defining condition for if condition
        p-=1
        print('Heuristic Value for Optimal Number of Lags :',p);break    #comes out of foor loop after the above if condition satisfies

# train auto regression  
model=AR(train.values)
model_fit = model.fit(p)

#make predictions
prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

rmse=((prediction-test)**2).mean()**0.5          #computing rmse value between predicted values and original value
print('\nRMSE for Optimal Lags : ',end='')
print(rmse)




#Solution to question 3 part d
print('\nd)')
print('\nWithout using Heuristics for Calculating Optimal Lag:')
print('\np:\t RMSE:')
optimal_index=RMSE.index(min(RMSE))      #finding minimum rmse 
print(Lag[optimal_index],'\t',RMSE[optimal_index])  #printing rmse and lag value corresponding to it

print('\nUsing Heuristics for Calculating Optimal Lag:')
print('\np:\t RMSE:')
print(p,'\t',rmse)   #prinitng rmse by using heuristics for calculating time lag
