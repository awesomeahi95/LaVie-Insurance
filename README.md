# LaVie Insurance

The primary goal of this project was to run and train statistical models and establish a clear process of understanding, training, validating, and testing the relationship between our dependent and independent variables.

## Business Objective:

As a struggling US life insurance company, our goal is to increase revenues by 2% using a risk premium based on premature deaths per state. We aim to hike insurance contract prices across the states that pause the highest future risk of premature death rates. This should in turn take our net premium growth rate above annual inflation which we have been on par with for the last five years and move to a more risk-adjusted business model which is key in our industry. 

## Data Structure and Selection: 

The dataset we used contained a wide ranging set of health ranking features per county within each state. This included data points such as premature deaths, low birthweight, adult smoking etc. As an intial step, we selected the premature death as our dependent variable, what we aim to predict. This variable is categorized as Years Potential Life Loff (YPLL) - estimate of the average years a person would have lived if he or she had not died prematurely. It is a measure of premature mortality. 

We then decided to exclude columns with confidence interval and quantile to simplify our selection process and model application. We subsequently ran a correlation matrix across all remaining variables and selected 31 variables with some form of positive or negative correlation and plausible causal relationsip with premature deaths. We purposefully kept a wide range of correlation selection as to avoid missing possible related interactions which could help us predict our dependent variable.

Some of our independent variables selected included: Adult Obesity, Food Environment Index, Physical Inactivity, Income inequality, Air Pollution.


## Transformation, Scaling, Modeling, Regularisation:

The initial step included setting up a structured framework to the data. We initially setup a train-test split at a 2/3 // 1/3 respective split to include at least 1000 observations in our testing test. We then performed 5-fold cross validation across our models for our training data to validate the completeness and quality of our data-model interaction. This also allowed us to select the highest performing models within each model group/type. 


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
