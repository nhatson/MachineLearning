from pandas import Series, DataFrame
import pandas as pd
import numpy as np
#import os
#import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#import sklearn.metrics
np.random.seed(11)

raw_data = pd.read_csv('data.csv',header = 0)
raw_data = pd.DataFrame(raw_data).fillna(0)
n = len(raw_data)
cleaned_data = raw_data.drop("CUSTOMER_NO",axis=1)
sLength = len(cleaned_data['SO_DU_HIEN_TAI'])
cleaned_data['e'] = Series(np.random.randn(sLength), index=cleaned_data.index)

for i in range(0, len(cleaned_data)):
	cleaned_data['e'][i] = cleaned_data['SO_DU_HIEN_TAI'][i]
print(cleaned_data['e'])

targets = cleaned_data.SO_DU_HIEN_TAI
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=0.2)
#print("Predictor — Training : ", pred_train.shape, "Predictor — Testing : ", pred_test.shape)

marginProfit = 3.52%
cost = 90000
d = 10%
def calculateProfit(amount):
	return marginProfit*amout - cost

def calculateCLV(amount1, amount2, amount3, amount4, amount5):
	profit1 = calculateProfit(amount1)
	profit2 = calculateProfit(amount2)
	profit3 = calculateProfit(amount3)
	profit4 = calculateProfit(amount4)
	profit5 = calculateProfit(amount5)
	return profit1/(1+d) + profit2/(1+d)**2 + profit3/(1+d)**3 + profit4/(1+d)**4 + profit5/(1+d)**5

CLV = []
for index in range(0, len(df)): 
	CLV.append(calculateCLV(amount1, amount2, amount3, amount4, amount5))

df.insert(loc=0, column='CLV', value = CLV)
