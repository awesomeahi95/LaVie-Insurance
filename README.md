# LaVie Insurance

## Business Objective:

As a struggling US life insurance company, our goal is to increase revenues by 2% using a risk premium based on premature deaths per state. We aim to hike insurance contract prices across the states that pause the highest future risk of premature death rates. This should in turn take our net premium growth rate above annual inflation which we have been on par with for the last five years and move to a more risk-adjusted business model which is key in our industry. 

## Data structure and selection: 

The dataset we used contained a wide ranging set of health ranking features per county within each state. This included data points such as premature deaths, low birthweight, adult smoking etc. As an intial step, we selected the premature death variable which is categorized as 

we decided to exclude columns with confidence interval and quantile to simplify our selection process and model application. 
We subsequently ran a correlation matrix across all remaining variables and selected 31 variables with some form of positive or negative correlation and plausible causal relationsip with premature deaths. 


## Findings:


## Model:


## Structure of Notebook:
1)Imports
    Importing Libraries
    Importing Dataset

2)EDA
    Early Exploration and Cleaning
    Testing Homoscedasticity
    Correlation Heatmap

3)Modelling
    Train Test Split
    Linear
    Quadratic
    Cubic
    Regularisation Using Regression Models

4)Validation
    Regularisation Using Regression Models
    Model Selection
    Coefficients of Best Models

5)Testing Final Model
    Lasso Model Test Data
    Coefficient Weighting
