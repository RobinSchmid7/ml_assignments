# Tasks

## Sub-task 1: Ordering of medical test

Binary classification:
- 0: no test
- 1: do test

Evaluation of performance with ROC AUC:
- ROC curve:
  - TPR vs. FPR at different classification thresholds
  - Lowering threshold classifies more items positive, hence increasing both FP and TP
  
- AUC:
  - measures area under ROC curve from (0,0) to (1,1)
  - scale-invariant: measures how well predictions are ranked
  - classification-threshold-invariant: measures quality of model's predictions
  
- https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
  
## Sub-task 2: Sepsis prediction

Binary classification:
- 0: no sepsis will occur
- 1: sepsis will occur

Evaluation the same as for sub-task 1.

## Sub-task 3: Key vital sign prediction

Regression task: predicting the means value of a vital sign for remaining stay

Evaluation of performance with R2 score (coefficient of determination).

# Approaches

- `isnull()` to check for missing values `nan
- idea to deal with time series data: unstack data to add new columns, 
  https://medium.com/swlh/reshaping-in-pandas-with-stack-and-unstack-functions-bb169f64467d
- dealing with missing data: only use features which have few missing values, use them unstacked
- for features which have few values only use mean value of all time steps and summarize them`

## fit_transform() and transform()

- `fit()` computes parameters and stores internally (standard scaler: zero mean and unit variance)
- calling `transform()` transforms every set using computed parameters (standard scaler: centering of data)
- `fit_transform()` joins both steps and is used when fitting training set, essentially calls `fit()` and then 
  `transform()`