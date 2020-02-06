
---
title = "LTFS Data Science FinHack 2"
---

## **Problem statement**

LTFS receives a lot of requests for its various finance offerings that include housing loan, two-wheeler loan, real estate financing and micro loans. The number of applications received is something that varies a lot with season. Going through these applications is a manual process and is tedious. Accurately forecasting the number of cases received can help with resource and manpower management resulting into quick response on applications and more efficient processing.

We have been appointed with the task of forecasting daily cases for **next 3 months for 2 different business segments** at the **country level** keeping in consideration the following major Indian festivals (inclusive but not exhaustive list): Diwali, Dussehra, Ganesh Chaturthi, Navratri, Holi etc. (We are free to use any publicly available open source external datasets). Some other examples could be:

 + Weather
 + Macroeconomic variables

we also note that the external dataset must belong to a reliable source.

## **Data Dictionary**

The train data has been provided in the following way:

 + For business segment 1, historical data has been made available at branch ID level
 + For business segment 2, historical data has been made available at State level.
 

## **Train File**

|Variable|	Definition|
|:------:|:----------:|
|application_date|Date of application|
|application_date|	Date of application|
|segment|	Business Segment (1/2)|
|branch_id|	Anonymised id for branch at which application was received|
|state|	State in which application was received (Karnataka, MP etc.)|
|zone|	Zone of state in which application was received (Central, East etc.)|
|case_count|	(Target) Number of cases/applications received|

## **Test File**

Forecasting needs to be done at country level for the dates provided in test set for each segment.

|Variable|	Definition|
|:------:|:----------:|
|id|	Unique id for each sample in test set|
|application_date|	Date of application|
| segment|	Business Segment (1/2)|

## **Evaluation**

**Evaluation Metric**

The evaluation metric for scoring the forecasts is MAPE (Mean Absolute Percentage Error) M with the formula:

$$M = \frac{100}{n}\sum_{t = 1}^{n}|\frac{A_t - F_t}{A_t}|$$
 
Where $A_t$ is the actual value and $F_t$ is the forecast value.


The Final score is calculated using $MAPE$ for both the segments using the formula:

$Final Score = 0.5*MAPE_{Segment1} + 0.5*MAPE_{Segment2}$


## **Getting started**

### **Importing libraries**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
%matplotlib inline

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
```


```python
# Setting the path
import os
path = "E:/Data Science/LTFS-Data-Science-FinHack-2"
os.chdir(path)
```

### **Reading data**


```python
# Reading data from github : train
#url_train = 'https://raw.githubusercontent.com/NiteshkumarJha/LTFS-Data-Science-FinHack-2/master/Input/train_fwYjLYX.csv'

#train = pd.read_csv(url_train)
```


```python
# Reading data from github : test
#url_test = 'https://raw.githubusercontent.com/NiteshkumarJha/LTFS-Data-Science-FinHack-2/master/Input/test_1eLl9Yf.csv'

#test = pd.read_csv(url_test)
```


```python
# Importing the dataset
train = pd.read_csv("./Input/train_fwYjLYX.csv")
test = pd.read_csv("./Input/test_1eLl9Yf.csv")
Sample_submission = pd.read_csv("./Input/sample_submission_IIzFVsf.csv")
```

## **Data Preprocessing**


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>application_date</th>
      <th>segment</th>
      <th>branch_id</th>
      <th>state</th>
      <th>zone</th>
      <th>case_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-04-01</td>
      <td>1</td>
      <td>1.00</td>
      <td>WEST BENGAL</td>
      <td>EAST</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-04-03</td>
      <td>1</td>
      <td>1.00</td>
      <td>WEST BENGAL</td>
      <td>EAST</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-04-04</td>
      <td>1</td>
      <td>1.00</td>
      <td>WEST BENGAL</td>
      <td>EAST</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-04-05</td>
      <td>1</td>
      <td>1.00</td>
      <td>WEST BENGAL</td>
      <td>EAST</td>
      <td>113.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-04-07</td>
      <td>1</td>
      <td>1.00</td>
      <td>WEST BENGAL</td>
      <td>EAST</td>
      <td>76.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Data preprocessing function
train_v2 = pd.DataFrame(train.groupby(['application_date', 'segment'])['case_count'].sum()).reset_index()
train_v2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>application_date</th>
      <th>segment</th>
      <th>case_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-04-01</td>
      <td>1</td>
      <td>299.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-04-01</td>
      <td>2</td>
      <td>897.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-04-02</td>
      <td>2</td>
      <td>605.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-04-03</td>
      <td>1</td>
      <td>42.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-04-03</td>
      <td>2</td>
      <td>2016.00</td>
    </tr>
  </tbody>
</table>
</div>



## **Feature engineering**


```python
def feature_eng(train_v2):
  train_v2['application_date'] = pd.to_datetime(train_v2['application_date'])
  train_v2['year'] = train_v2['application_date'].dt.year
  train_v2['Month'] = train_v2['application_date'].dt.month
  train_v2['Date'] = train_v2['application_date'].dt.day
  train_v2['weekday'] = train_v2['application_date'].dt.weekday_name

  Seasons = {6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
           10: 'Winter', 11: 'Winter', 12: 'Winter', 1: 'Winter',
           2: 'Summer', 3: 'Summer', 4: 'Summer', 5: 'Summer'}
  
  train_v2['Seasons'] = train_v2['Month'].map(Seasons)

  train_v2['segment'] = np.where(train_v2['segment'] == 1, 1, 0)

  dummy_col = ['weekday', 'Seasons']
  temp = train_v2[dummy_col]
  temp = pd.get_dummies(temp)

  train_v2 = train_v2.drop(dummy_col, axis = 1)
  train_v2 = pd.concat([train_v2, temp], axis = 1)

  train_v2 = train_v2.drop(['application_date'], axis = 1)
  
  return train_v2
```

## **Machine Learning**

### **Creating X and y**


```python
X = train_v2.drop(['case_count'], axis = 1)
y = np.log(train_v2['case_count'])

X = feature_eng(X)

print("Shape of features :", X.shape)
print("Shape of labels :", y.shape)

X.head()
```

    Shape of features : (1650, 14)
    Shape of labels : (1650,)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>segment</th>
      <th>year</th>
      <th>Month</th>
      <th>Date</th>
      <th>weekday_Friday</th>
      <th>weekday_Monday</th>
      <th>weekday_Saturday</th>
      <th>weekday_Sunday</th>
      <th>weekday_Thursday</th>
      <th>weekday_Tuesday</th>
      <th>weekday_Wednesday</th>
      <th>Seasons_Monsoon</th>
      <th>Seasons_Summer</th>
      <th>Seasons_Winter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2017</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2017</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2017</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2017</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2017</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### **Splitting data into train, validation and test**


```python
# Dividing data into train and validation set
from sklearn.model_selection import train_test_split

validation_percent = 0.30
test_percent = 0.50
seed = 786

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = validation_percent, random_state = seed)
X_validation, X_test, y_validation, y_test = train_test_split(X_validation, y_validation, test_size = test_percent, random_state = seed)

# Shape of data
print("Number of rows and columns in train dataset:",X_train.shape)
print("Number of rows and columns in validation dataset:",X_validation.shape)
print("Number of rows and columns in test dataset:",X_test.shape)

print("Number of rows and columns in target variable for training:",y_train.shape)
print("Number of rows and columns in target variable for validation:",y_validation.shape)
print("Number of rows and columns in target variable for test:",y_test.shape)
```

    Number of rows and columns in train dataset: (1155, 14)
    Number of rows and columns in validation dataset: (247, 14)
    Number of rows and columns in test dataset: (248, 14)
    Number of rows and columns in target variable for training: (1155,)
    Number of rows and columns in target variable for validation: (247,)
    Number of rows and columns in target variable for test: (248,)
    

### **Model evualuation**


```python
import sklearn.metrics as sklm
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, PassiveAggressiveRegressor, Perceptron
from sklearn.neighbors import KNeighborsRegressor, NearestCentroid
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor 
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
```


```python
def mape(forecast, actual):
  mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
  return mape
```


```python
def accuracy_summary(Regressor, x_train, y_train, x_validation, y_validation):
    t0 = time()
    model = Regressor.fit(x_train, y_train)
    y_pred = model.predict(x_validation)
    train_test_time = time() - t0
    #accuracy = r2_score(y_validation, y_pred)
    accuracy = mape(y_pred, y_validation)
    return accuracy, train_test_time
```


```python
seed = 123
names = ["Linear Regression", "SGDRegressor", "Linear SVR", "Lasso","Ridge", "Passive-Aggresive",
        "DecisionTreeRegressor","RandomForestRegressor","AdaBoostRegressor", "GradientBoostingRegressor", "XGBRegressor"]

Regressors = [
    LinearRegression(),
    SGDRegressor(random_state=seed),
    LinearSVR(random_state=seed),
    #SVR(),
    Lasso(random_state=seed),
    Ridge(random_state=seed),
    PassiveAggressiveRegressor(random_state=seed),
    DecisionTreeRegressor(random_state=seed),
    RandomForestRegressor(random_state=seed, n_estimators=500),
    AdaBoostRegressor(random_state=seed, n_estimators=500),
    GradientBoostingRegressor(random_state=seed, n_estimators=500),
    XGBRegressor(n_estimators=500, random_state=seed)
    ]

zipped_reg = zip(names,Regressors)

def Regressor_comparator(Regressor=zipped_reg):
    result = []
    for n,c in Regressor:
        checker_pipeline = Pipeline([
            ('Regressor', c)
        ])
        print("Validation result for {}".format(n))
        print (c)
        reg_accuracy,tt_time = accuracy_summary(checker_pipeline, X_train, y_train, X_validation, y_validation)
        result.append((n,reg_accuracy,tt_time))
    return result
```


```python
Regression_result = Regressor_comparator()
Regression_result
```

    Validation result for Linear Regression
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    Validation result for SGDRegressor
    SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
                 eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                 learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                 n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=123,
                 shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
                 warm_start=False)
    Validation result for Linear SVR
    LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
              intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
              random_state=123, tol=0.0001, verbose=0)
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    

    Validation result for Lasso
    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
          normalize=False, positive=False, precompute=False, random_state=123,
          selection='cyclic', tol=0.0001, warm_start=False)
    Validation result for Ridge
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=123, solver='auto', tol=0.001)
    Validation result for Passive-Aggresive
    PassiveAggressiveRegressor(C=1.0, average=False, early_stopping=False,
                               epsilon=0.1, fit_intercept=True,
                               loss='epsilon_insensitive', max_iter=1000,
                               n_iter_no_change=5, random_state=123, shuffle=True,
                               tol=0.001, validation_fraction=0.1, verbose=0,
                               warm_start=False)
    Validation result for DecisionTreeRegressor
    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=123, splitter='best')
    Validation result for RandomForestRegressor
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=500,
                          n_jobs=None, oob_score=False, random_state=123, verbose=0,
                          warm_start=False)
    Validation result for AdaBoostRegressor
    AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
                      n_estimators=500, random_state=123)
    Validation result for GradientBoostingRegressor
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                              learning_rate=0.1, loss='ls', max_depth=3,
                              max_features=None, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=500,
                              n_iter_no_change=None, presort='auto',
                              random_state=123, subsample=1.0, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)
    Validation result for XGBRegressor
    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                 max_depth=3, min_child_weight=1, missing=None, n_estimators=500,
                 n_jobs=1, nthread=None, objective='reg:linear', random_state=123,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                 silent=True, subsample=1)
    




    [('Linear Regression', 0.10513573449395285, 1.731126070022583),
     ('SGDRegressor', 60169262908319.36, 0.12459874153137207),
     ('Linear SVR', 0.3326559027917069, 0.25212812423706055),
     ('Lasso', 0.15777142146560802, 0.16437602043151855),
     ('Ridge', 0.10512046272408691, 0.3259265422821045),
     ('Passive-Aggresive', 0.17484383055544325, 0.01399683952331543),
     ('DecisionTreeRegressor', 0.08117820373237314, 0.0381016731262207),
     ('RandomForestRegressor', 0.05420337132096428, 2.819838762283325),
     ('AdaBoostRegressor', 0.09231015135427076, 0.09985017776489258),
     ('GradientBoostingRegressor', 0.07042529886880469, 0.6586427688598633),
     ('XGBRegressor', 0.07042746358188312, 0.6774027347564697)]




```python
Regression_result_df = pd.DataFrame(Regression_result)
Regression_result_df.columns = ['Regressor', 'R2-Score', 'Train and test time']
Regression_result_df.sort_values(by='R2-Score', ascending=False)
Regression_result_df['R2-Score'] = (Regression_result_df['R2-Score']*100).round(1).astype(str) + '%'
Regression_result_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Regressor</th>
      <th>R2-Score</th>
      <th>Train and test time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>10.5%</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SGDRegressor</td>
      <td>6016926290831936.0%</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Linear SVR</td>
      <td>33.3%</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lasso</td>
      <td>15.8%</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ridge</td>
      <td>10.5%</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Passive-Aggresive</td>
      <td>17.5%</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DecisionTreeRegressor</td>
      <td>8.1%</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RandomForestRegressor</td>
      <td>5.4%</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AdaBoostRegressor</td>
      <td>9.2%</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GradientBoostingRegressor</td>
      <td>7.0%</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>10</th>
      <td>XGBRegressor</td>
      <td>7.0%</td>
      <td>0.68</td>
    </tr>
  </tbody>
</table>
</div>



### **Tuning Randomforest model**


```python
model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=500,
                      n_jobs=None, oob_score=False, random_state=123, verbose=0,
                      warm_start=False)

model = model.fit(X_train, y_train)
y_predict = model.predict(X_validation)
```


```python
import math
def print_metrics(y_true, y_predicted):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    MAPE = mape(y_predicted, y_true)
    # r2_adj = r2 - (y_true.shape[0] - 1)/(y_true.shape[0] - n_parameters - 1) * (1 - r2)
    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('MAPE                    = ' + str(MAPE))
    
    # print('Adjusted R^2           = ' + str(r2_adj))

print_metrics(y_validation, y_predict)
```

    Mean Square Error      = 0.3008800222513618
    Root Mean Square Error = 0.5485253159621366
    Mean Absolute Error    = 0.28422202650153067
    Median Absolute Error  = 0.12658874394035635
    R^2                    = 0.8364501577346957
    MAPE                    = 0.05420337132096428
    

## **Predicting test data**


```python
test_v2 = test.drop(['id'], axis = 1)
test_v2 = feature_eng(test_v2)

print("Shape of features :", test_v2.shape)
```

    Shape of features : (180, 13)
    


```python
feature_list = X.columns.tolist()
for 
feature_list
```


      File "<ipython-input-18-bc7a2005f352>", line 2
        for
            ^
    SyntaxError: invalid syntax
    

