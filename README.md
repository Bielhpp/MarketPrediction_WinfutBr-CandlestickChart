# Mini Index Prediction Project (WINFUT - CandlestickChart)
This repository contains the implementation of a Mini Index prediction project using data extracted through the Nelógica ProfitPro program, using market replay. The project aims to explore different approaches and prediction models, with at least seven planned versions to assess the system's performance and refinement over time.

## General Description
**Database**
The database used contains up to 10,000 records, the maximum allowed extraction per view in ProfitPro. 400 records were excluded due to missing data in moving average indicators (400 periods).

**First Version of the Model**
The first version of the model is based on the project described in dataquestio/project-walkthroughs, to which we refer.

**Code and Features**
The code is implemented in Python using the Pandas library for data manipulation and scikit-learn for building and evaluating machine learning models. Below is an overview of the code:

**1- Data Reading and Pre-processing**

import pandas as pd
import os

winfutcandle5min = pd.read_excel('winfutcandle5min.xlsx')
winfutcandle5min['Data'] = pd.to_datetime(winfutcandle5min['Data'], dayfirst=True) 
winfutcandle5min = winfutcandle5min.sort_values(by='Data', ascending=True)
winfutcandle5min = winfutcandle5min.set_index('Data')

**2- Data Visualization**

winfutcandle5min.plot.line(y="Fechamento", use_index=True)

**3- Model Creation and Training (Random Forest)**

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
train = winfutcandle5min.iloc[:-100]
test =winfutcandle5min.iloc[-100:]
predictors = ["Fechamento", "Volume Quantidade", "Abertura", "Máxima", "Mínima"]
model.fit(train[predictors], train["Alvo"])

**4- Model Evaluation**

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
precision = precision_score(test["Alvo"], preds)
print(f"Precision Score: {precision}")

**5- Backtesting and Results Analysis**

predictions = backtest(winfutrenko, model, predictors)
precision_backtest = precision_score(predictions["Alvo"], predictions["Previsoes"])
print(f"Precision Score (Backtest): {precision_backtest}")

**6- Subsequent Model Versions**

Subsequent versions of the model were explored with new predictors, including moving averages and trends.

## Final Considerations
This project is a work in progress, and future versions will be implemented to enhance prediction accuracy. Your contributions and suggestions are welcome.

Thanks to the reference to the dataquestio/project-walkthroughs project that served as inspiration for the first version.
