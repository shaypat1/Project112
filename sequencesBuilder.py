import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data.csv")

data['ClosePercentChange'] = data["Close"].pct_change()
data['VolumePercentChange'] = data['Volume'].pct_change()

#Standardize
scaler = StandardScaler()
data["StandardClose"] = scaler.fit_transform(data[['Close']])
data['StandardVolume'] = scaler.fit_transform(data[['Volume']])

def makeSequences(features, target, seqLength = 10):
    xSeq = []
    ySeq = []
    #data.iloc (dataframe slicing)
    for i in range(len(features)-seqLength-1):
        x = features.iloc[i:(i+seqLength)].values #10 values
        y = target.iloc[i+seqLength] #11th value
        xSeq.append(x)
        ySeq.append(y)
    return np.array(xSeq), np.array(ySeq)



tickerSeq = dict()
for ticker in data['Symbol'].unique():
    # use boolean selection
    tickerData = data[data["Symbol"]==ticker].copy()
    features = tickerData[['StandardClose', "StandardVolume", 'ClosePercentChange', 'VolumePercentChange']]
    tickerData['Target'] = tickerData['StandardClose'].shift(-1)>tickerData['StandardClose'].astype(int)
    target = tickerData['Target']
    sequences, results = makeSequences(features, target)
    tickerSeq[ticker] = (sequences, results)

