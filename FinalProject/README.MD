## Description of Model 
The following is an overview of the bagging classifier model created to predict the market direction of a given currency pair. The model was designed to run on the QuantConnect trading platform. 

### Feature Engineering
The following lists the features that we engineered for this model:
1. Variables relating to candlestick approach: upper shadow, lower shadow, body
- Body = Close - Open
- Lower Shadow = Min(Open, Close) - Low
- Upper Shadow = High - Max(Open, Close)
2. Momentum analysis, by deriving the 5-day moving average and the 15-day moving average  
3. Lagged prices of the USD/ZAR over the last 5 days
The former two features are technical indicators that are used in traditional analysis of forex. Different patterns found in the candlestick chart can be used to predict whether the direction of trend is going to change. Additionally, the relative position of 5-day moving average and 15-day moving average can predict the direction the price is going. For example, when the 5-day moving average crosses above the 15-day moving average, this indicates that price is rising and hence conveys a buy signal to the model. Similarly, the lagged prices of the USD/ZAR is a good feature as price at one point in time is naturally correlated to the neighbouring prices in that period of time. 

### Execution of Trade & Risk Management
Based on the various features, the model will predict whether the closing price will go “UP” or “DOWN” on the following day and the model uses this as a trade signal: The model will short if price is predicted to go down and will long if price is predicted to go up. The model will check whether the trade condition is met every 20 minutes and will take profit and stop loss accordingly. The Stop Loss Take Profit (SLTP) condition will be triggered when the price deviates 1.5% away from cost basis . Since the model has the same stop loss and take profit margin, it has 1:1 risk/reward ratio. While we have considered higher risk/reward ratio, a poorer backtest performance deters us from using a higher risk/reward ratio and thus we have chosen to use a very conservative risk/reward ratio.
  

