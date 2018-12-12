import numpy as np
from numpy.random import seed

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta 

import pandas as pd 


#variables 
Symbol = "USDZAR" 
Brokerage = BrokerageName.InteractiveBrokersBrokerage 
capital = 100000 
AlgoResolution = Resolution.Hour  
TimePeriod = 50  
LongStopLoss = 0.985
LongTakeProfit = 1.015
ShortStopLoss = 1.015
ShortTakeProfit = 0.985 


class Forex_Trade_Logreg(QCAlgorithm):
    
#####  17 to 34:  Initialization of Algo ####
    def Initialize(self):
        
        self.Debug('Stop Loss is ' + str(LongStopLoss))
        self.Debug('Take Profit is ' + str(LongTakeProfit)) 
        self.Debug('algo resolution is ' + str(AlgoResolution)) 
        
        #self.Debug("START: Initialize")
        self.SetStartDate(2018,11,7)    #Set Start Date
        self.SetEndDate(2018,11,21)     #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin) #Set Brokerage Model
        self.currency = Symbol 
        self.AddForex(self.currency,Resolution.Daily)
        self.long_list = []
        self.short_list = []
        self.model = BaggingClassifier(n_estimators=100, random_state=12345)
        #self.model = LogisticRegression()
        self.x = 0
        self.output = ''  
        self.currency_data = None 
        #self.Debug("End: Initialize")
        
       

#####  37 to 48 : Defining OnData function and Geting the Historical Data  ####
    def OnData(self, data): #This function runs on every resolution of data mentioned. 
                            #(eg if resolution = daily, it will run daily, if resolution = hourly, it will run hourly.)
        
        #self.Debug("START: Ondata")
        self.currency_data  = self.History([self.currency], 500, Resolution.Daily) # Asking for historical data 
        #self.Debug('self.currency_data is ' + str(self.currency_data)) 
        
        
        L = len(self.currency_data) # Checking the length of data
        #self.Debug("The length is " + str (L))
    
#####  52 to 81 : Check condition for required data and prepare X and Y for modeling  ####    
        # Feature 1: body
        self.currency_data["body"] = self.currency_data["close"] - self.currency_data["open"]
        self.currency_data["oribody"] = self.currency_data["body"]
        self.currency_data["body"] = self.currency_data["body"].shift(1) # lag one day
        
        # Feature 2: upper shadow
        self.currency_data["upshadow"] = self.currency_data.apply(lambda x: x['high'] - x['open'] if x['body'] < 0 else x['high'] - x['close'], axis = 1)
        self.currency_data["oriupshadow"] = self.currency_data["upshadow"]
        self.currency_data["upshadow"] = self.currency_data["upshadow"].shift(1) # lag one day
        
        # Feature 3: lower shadow
        self.currency_data["lowshadow"] = self.currency_data.apply(lambda x: x['close'] - x['low'] if x['body'] < 0 else x['open'] - x['low'], axis = 1)
        self.currency_data["orilowshadow"] = self.currency_data["lowshadow"]
        self.currency_data["lowshadow"] = self.currency_data["lowshadow"].shift(1) # lag one day
            
        # Function for simple moving average
        def sma(df, days, row, col):
            result = 0
            
            if row < days:
                return result
            
            else:
                for i in range(row-days, row+1):
                    result += df[col].iloc[i]
                result /= days
                return result
        
        # Feature 4 & 5: simple moving average    
        for x in range(len(self.currency_data)):
            self.currency_data.loc[self.currency_data.index[x], "closesma5"] = sma(self.currency_data, 5, x, "close")
            self.currency_data.loc[self.currency_data.index[x], "closesma15"] = sma(self.currency_data, 15, x, "close")
            
        # Feature 6: lag prices
        self.currency_data["lag1"] = self.currency_data.close.shift(1)
        self.currency_data["lag2"] = self.currency_data.close.shift(2)
        self.currency_data["lag3"] = self.currency_data.close.shift(3)
        self.currency_data["lag4"] = self.currency_data.close.shift(4)
        self.currency_data["lag5"] = self.currency_data.close.shift(5)
       
        stored = self.currency_data[['close', 'closesma5', 'closesma15', 'body', 'upshadow', 'lowshadow', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5']]
        stored = pd.DataFrame(stored)
        
        stored = stored.dropna() # drop na values
        stored = stored.reset_index(drop=True)
        
        corelation = stored.corr()
        #self.Debug("corr is" +str(corelation))
        stored["Y"] = stored["close"].pct_change() # get the percent change from previous time
        
        for i in range(len(stored)): # loop to make Y as categorical
            if stored.loc[i,"Y"] > 0:
                stored.loc[i,"Y"] = "UP"
            else:
                stored.loc[i,"Y"] = "DOWN"
                    
        #self.Debug("All X_data is " +str(stored))    
        
        X_data = stored[['closesma5', 'closesma15', 'body', 'upshadow', 'lowshadow', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5']]
        #self.Debug( "X data is" +str(X_data))
        
        Y_data = stored["Y"]
        #self.Debug( "Y data is" +str(Y_data))
            
#####  85 to 97 : Build the Logistic Regression model, check the training accuracy and coefficients  ####     
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.3, random_state = 12345)

        if self.x==0:  #To make sure the model is build only once and avoid computation at every new data
            
            self.model.fit(X_train,y_train)
            score = self.model.score(X_test, y_test)
            self.Debug("Train Accuracy of final model: " + str(score))
                
            self.x=1     # End the model 

#####  102 to 110 : Prepare data for prediction   ####            
        lst = []
        
        def smatest(df, x, col):
            total = 0
            for i in range(1, x+1):
                total += df.iloc[len(df)-i, col]
            return total/x
        
        lst.append(smatest(self.currency_data, 5, 0)) #closesma5
        lst.append(smatest(self.currency_data, 15, 0)) #closesma15
        
        lst.append(self.currency_data.ix[len(self.currency_data)-1, "oribody"]) #body
        lst.append(self.currency_data.ix[len(self.currency_data)-1, "oriupshadow"]) #upshadow
        lst.append(self.currency_data.ix[len(self.currency_data)-1, "orilowshadow"]) #lowshadow
        
        lst.append(self.currency_data.iloc[-1, 0]) #lag1
        lst.append(self.currency_data.iloc[-2, 0]) #lag2
        lst.append(self.currency_data.iloc[-3, 0]) #lag3
        lst.append(self.currency_data.iloc[-4, 0]) #lag4
        lst.append(self.currency_data.iloc[-5, 0]) #lag5
       
        self.Debug(lst)
        test = pd.DataFrame([lst], columns = ['closesma5', 'closesma15', 'body', 'upshadow', 'lowshadow', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5'])
        #self.Debug(test)


#####  115 to 116 : Make Prediction   #### 
        self.output = self.model.predict(test)

        #Checking the current price 
        price = self.currency_data.close[-1]
        
        #Make decision for trading based on the output from LR and the current price.
        #If output ( forecast) is UP we will buy the currency; else, Short.
        # Only one trade at a time and therefore made a list " self.long_list". 
        #As long as the currency is in that list, no further buying can be done.
        # Risk and Reward are defined: Ext the trade at 1% loss or 1 % profit.


    #self.Debug('enter check price function') 
    
    #check the time that the code is fired at 
    # self.Debug("SpecificTime: Fired at : {0}".format(self.Time))

        if self.x == 1: 
            #Checking the current price  
            #self.Debug("check price!") 
            self.Debug('output is ' + str(self.output)) 
            price = self.currency_data.close[-1]
            
            if self.output == "DOWN"  and self.currency not in self.long_list:
                
                #self.Debug("output is greater")
                # Buy the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, 0.9)
                self.long_list.append(self.currency)
                self.Debug("long")
                self.Debug('long list is ' + str(self.long_list)) 
                
            if self.currency in self.long_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(LongStopLoss) * float(cost_basis)) or (price >= float(LongTakeProfit) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then sell
                    self.SetHoldings(self.currency, 0)
                    self.long_list.remove(self.currency)
                    self.Debug("squared long")
                    
                    
            if self.output =="UP" and self.currency not in self.short_list:
                
                #self.Debug("output is lesser")
                # Buy the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, -0.9)
                self.short_list.append(self.currency)
                self.Debug("short") 
                
            if self.currency in self.short_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(ShortTakeProfit) * float(cost_basis)) or (price >= float(ShortStopLoss) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then buy back
                    self.SetHoldings(self.currency, 0)
                    self.short_list.remove(self.currency)
                    self.Debug("squared short")
            #self.Debug("END: Ondata") 
 
