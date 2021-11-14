# Stock Price Predictor
 
## Basis

We assume that the intrinsic value of a stock is dependent on a number of factors such as P/E ratio, EBITDA, cashflow, etc and an estimate of the price of the stock is then a function of the values of these factors. Therefore, we can use these values as inputs to train a ML model to predict the intrinsic value of a stock.

We further assume that the market price of a stock generally reflects the intrinsic values. This means that we can train the ML model with existing stocks, and stocks for which the model predicts a much lower price than the market price are effectively under-priced and the reverse is true for over-pricing.

These assumptions are obviously over-simplifications, but assumed for the purposes of this project.

## Settings

We use the yfinance library https://github.com/ranaroussi/yfinance to obtain fundamental data for stocks. The data collected reflects all stocks in NYSE and NASDAQ exchanges for which yfinance has adequate data. The data collected is stored in (stock_data.csv).

We use tensorflow to build the machine learning model.
