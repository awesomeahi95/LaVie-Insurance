# LaVie Insurance

The primary goal of this project was to utilise machine learning methods including regression and regularisation to find the best predictive model to understand the mulitvariate predictors of premature death rate, our proxy for life insurance premiums.

## Table of Contents

1. [ File Descriptions ](#file_description)
2. [ Strucuture ](#structure)
3. [ Executive Summary ](#executive_summary)
   * [ Data Structure and Selection ](#data_structure_and_selection)
   * [ Transformation, Scaling, Modelling, Regularisation ](#modelling)
   * [ Conclusion ](#conclusion)

<a name="file_description"></a>
## File Descriptions

- index.ipynb: notebook with data exploration, preprocessing, and modelling.
- init.py: contains classes for linear regression, polynomials, and regularisation.
- analytic_data2019.csv: data used for modelling.
- model.pkl: final model saved using pickle.
- Regression and Insurance.pdf: presentation summarising project process and findings

<a name="structure"></a>
## Structure of Notebook:
1. Imports
   - Importing Libraries
   - Importing Dataset

2. EDA
   - Early Exploration and Cleaning
   - Testing Homoscedasticity
   - Correlation Heatmap

3. Modelling
   - Train Test Split
   - Linear
   - Quadratic
   - Cubic
   - Regularisation Using Regression Models

4. Validation
   - Regularisation Using Regression Models
   - Model Selection
   - Coefficients of Best Models

5. Testing Final Model
   - Lasso Model Test Data
   - Coefficient Weighting

<a name="executive_summary"></a>
## Executive Summary

As a struggling US life insurance company, our goal is to increase revenues by 2% using a risk premium based on premature deaths per state. We aim to hike insurance contract prices across the states that pause the highest future risk of premature death rates. This should in turn take our net premium growth rate above annual inflation which we have been on par with for the last five years and move to a more risk-adjusted business model which is key in our industry. 

<a name="data_structure_and_selection"></a>
### Data Structure and Selection: 

The dataset we used contained a wide ranging set of health ranking features per county within each state. This included data points such as premature deaths, low birthweight, adult smoking etc. As an intial step, we selected the premature death as our dependent variable, what we aim to predict. This variable is categorized as Years Potential Life Loff (YPLL) - estimate of the average years a person would have lived if he or she had not died prematurely. It is a measure of premature mortality. 

We then decided to exclude columns with confidence interval and quantile to simplify our selection process and model application. In addition, we tested for multicollinearity for the independent variables using a VIF (variance inflation factor) test. It yield some highly collinear relationships across variables that we then excluded. 

We subsequently ran a correlation matrix across all remaining variables and selected 31 variables with some form of positive or negative correlation and plausible causal relationsip with premature deaths. We purposefully kept a wide range of correlation selection as to avoid missing possible related interactions which could help us predict our dependent variable.

Some of our independent variables selected included: Adult Obesity, Food Environment Index, Physical Inactivity, Income inequality, Air Pollution.

<h5 align="center">Correlation Heatmap</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/LaVie-Insurance/Correlation_Heatmap.png" width=850>
</p>

<a name="modelling"></a>
### Transformation, Scaling, Modelling, Regularisation:

The initial step included setting up a structured framework to the data. We initially setup a train-test split at a 2/3 // 1/3 respective split to include at least 1000 observations in our testing test. We then performed 5-fold cross validation across our models for our training data to validate the completeness and quality of our data-model interaction. This also allowed us to select the highest performing models within each model group/type. 

The initial baseline model was a multivariate linear regression using the Ordinary Least Squares (OLS) method. With 31 different variables the model yielded an R^2 score of 0.6154 which was an encouraging sign as an first instance model.

We then proceeded to perform a polynomial transformation to our variables to factor interactions of independent variables and their explanatory power of premature deaths. We ran three levels of polynomial regression: quadratic, qubic and quartic. The best results came out of the quadratic transformation with an improved R^2 score of 0.7610. We elected to keep this transformation.  

Using a z-score standard scaler method we took our poly-transformed data and scaled it before applying the regularisation methods. 

To improve our coefficients and simplify our model we decided to run a series of regularisation tests using lasso, ridge and elastic net regressions with several alpha levels. We once again applied the 5-fold validation process to find the best R^2 value across model types with the optimal level. Our rationale at this point is to come back to our stakeholders and make sure the regression is as simple as possible to apply it in a business context. We selected a lasso model yielding an R^2 of 0.6454811 on the training set with an alpha level of 27. 

Finally, we ran the model on the test set and got an R^2 of 0.681107 illustrating some accuracy in our model and a possibility to deploy it for our business application. 

<h5 align="center">Number of Fights in the UFC Each Year</h5>
<p align="center">
  <img src="https://github.com/ravimalde/ufc_fight_predictor/blob/master/images/number_of_fights.png" width=850>
</p>

<a name="conclusion"></a>
### Conclusion

We believe our final model provides us enough prediction accuracy to implement the life insurance premium increase. We are aiming to perform this increase as a function of YPLL predictions scaled to the population size. This model should allow us not only to increase our revenues but to shift towards a more risk-adjusted revenue approach which we can replicate in the future. 

