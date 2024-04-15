import yfinance as yf
import pandas as pd
import json

with open("data.json") as f:
    data = json.loads(f.read())

# Create a list to hold individual dataframes
tableData = []

# Add each individual data frame for each ticker

for ticker in data:
    stock = yf.Ticker(ticker)
    stockData = stock.history(period = '1mo', start='2020-1-1', end='2023-12-31')
    stockData["Symbol"] = ticker
    tableData.append(stockData)

# Combine dataframe 
data = pd.concat(tableData)




data.to_csv('data.csv')

