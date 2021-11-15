# Stock Price Predictor
 
## Basis

We assume that the intrinsic value of a stock is a function of the values of fundamental factors such as P/E ratio, EBITDA, cashflow, etc. Therefore, we can use these values as inputs to train a ML model to estimate the intrinsic value of a stock.

We further assume that market prices of stocks generally reflects the intrinsic values when averaged over the a large number of stocks, i.e. the market prices stocks accurately for the vast majority of stocks. 

This suggests that if we train the ML model with a sufficiently large dataset, the stocks for which the model predicts a significantly lower price than the market price are effectively under-priced by the market.

These assumptions are obviously over-simplifications, but assumed for the purposes of this project. We can use this predictor as a screening tool.

## Settings

We use the yfinance library https://github.com/ranaroussi/yfinance to obtain fundamental data for stocks. The data collected reflects all stocks in NYSE and NASDAQ exchanges for which yfinance has adequate data. The data collected is stored in (stock_data.csv).

We use pandas to process the data and generate the input vectors according to the factors in (stock_intrinsic_factors.json). The input vectors are normalised to have mean = 0 and variance = 1.

We create the model in TensorFlow and train it using the mean squared error as the loss function. The output is the predicted price of a stock.

## Usage

Usage demonstration is presented in (stock_price_predictor.py).
