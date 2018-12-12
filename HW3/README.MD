### BT2101 HW3
Cryptocurrency trading algorithm to run on QuantConnect platform.

The output price of the selected currency pair ("BTCUSD") was predicted based on the closing price of the previous 3 days. Using this data, we trained a LTSM model, a decision tree regressor and a linear regression model. For simplicity sake, we assumed the predictions for each model to be of equal weightage. Thus, we simply took the average of each model to get the aggregated prediction. 
