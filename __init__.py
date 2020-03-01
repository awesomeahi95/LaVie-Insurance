#===========================================================================================================================================================
# Imports
#===========================================================================================================================================================
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.model_selection import cross_validate
import warnings 
from __init__ import *
warnings.filterwarnings('ignore')

#===========================================================================================================================================================
# Basline Regression Models
#===========================================================================================================================================================

class Baseline:
    def __init__(self,x,y,kfolds):
        self.x = x
        self.y = y
        self.kfolds = kfolds
    
    def get_score(self):
        '''
        Train and Validate Linear Regression Models.
        '''
        baseline_model = LinearRegression().fit(self.x,self.y)
        self.coef_values = (pd.DataFrame(baseline_model.coef_)).T
        self.coef_values.columns = self.x.columns
        model_score = cross_validate(baseline_model,
                                     self.x,
                                     self.y, 
                                     scoring=('r2', 'neg_mean_squared_error'),
                                     cv=self.kfolds,
                                     return_train_score=True)
        self.r2valid_score = round(np.mean(model_score['test_r2']),4)
        self.r2train_score = round(np.mean(model_score['train_r2']),4)
        self.r2difference = round(abs(self.r2train_score-self.r2valid_score),4)
        self.msevalid_score = round(np.mean(model_score['test_neg_mean_squared_error']),-4)
        self.msetrain_score = round(np.mean(model_score['train_neg_mean_squared_error']),-4)
        self.msedifference = round(abs(self.msetrain_score-self.msevalid_score),-4)
        self.score = pd.DataFrame({"R^2 Train Score":[self.r2train_score],"R^2 Validation Score":[self.r2valid_score],
                                   "R^2 Difference":[self.r2difference],"MSE Train Score":[self.msetrain_score],
                                   "MSE Validation Score":[self.msevalid_score],"MSE Difference":[self.msedifference],
                                   "Number of Coefficients":[len(baseline_model.coef_)]})
        return self.score
    
#===========================================================================================================================================================
# Polynomial Regression Models
#===========================================================================================================================================================

class Polynomials():
    def __init__(self,x,y,kfolds,degree):
        self.x = x
        self.y = y
        self.kfolds = kfolds
        self.degree = degree
        self.coef_values = pd.DataFrame()
        self.column_names = pd.DataFrame()
    
    def get_score(self):
        '''
        Train and Validate Polynomial Regression Models.
        '''
        poly_reg = PolynomialFeatures(self.degree)
        x_poly = pd.DataFrame(poly_reg.fit_transform(self.x))
        reg_poly = LinearRegression().fit(x_poly,self.y)
        self.column_names = poly_reg.get_feature_names(self.x.columns)
        self.coef_values = (pd.DataFrame(reg_poly.coef_)).T
        self.coef_values.columns = self.column_names
        self.x_poly = x_poly
        self.x_poly.columns = self.column_names
        self.coef_values.drop(["1"],axis=1,inplace=True)
        model_score = cross_validate(reg_poly,
                                     x_poly,
                                     self.y, 
                                     scoring=('r2', 'neg_mean_squared_error'),
                                     cv=self.kfolds,
                                     return_train_score=True)
        self.r2valid_score = round(np.mean(model_score['test_r2']),4)
        self.r2train_score = round(np.mean(model_score['train_r2']),4)
        self.r2difference = round(abs(self.r2train_score-self.r2valid_score),4)
        self.msevalid_score = round(np.mean(model_score['test_neg_mean_squared_error']),-4)
        self.msetrain_score = round(np.mean(model_score['train_neg_mean_squared_error']),-4)
        self.msedifference = round(abs(self.msetrain_score-self.msevalid_score),-4)
        self.score = pd.DataFrame({"R^2 Train Score":[self.r2train_score],"R^2 Validation Score":[self.r2valid_score],
                                   "R^2 Difference":[self.r2difference],"MSE Train Score":[self.msetrain_score],
                                   "MSE Validation Score":[self.msevalid_score],"MSE Difference":[self.msedifference],
                                   "Number of Coefficients":[len(self.coef_values.columns)]})
        return self.score
    
    def poly_reg_test(self,X_test):
        poly_reg = PolynomialFeatures(self.degree)
        X_poly = pd.DataFrame(poly_reg.fit_transform(self.x))
        X_poly_test = pd.DataFrame(poly_reg.transform(X_test))
        return X_poly_test

#===========================================================================================================================================================
# Regularisation Regression Models
#===========================================================================================================================================================

class Regularisation:
    def __init__(self,reg_type,x,y,kfolds,alpha_max):
        self.reg_type = reg_type
        self.x = x
        self.y = y
        self.kfolds = kfolds
        self.alpha_max = alpha_max
        self.column_names = self.x.columns
        
        self.r2valid_score = []
        self.r2train_score = []
        self.r2difference = []
        self.msevalid_score = []
        self.msetrain_score = []
        self.msedifference = []
        self.alphalist = []
        self.all_coefficients= []
        self.non_zero_coefficients = []
        self.len_all_coefs = []
        
        if reg_type == 'lasso':
            self.technique = Lasso
        elif reg_type == 'ridge':
            self.technique = Ridge
        elif reg_type == 'elastic':
            self.technique = ElasticNet

    def get_scores(self):
        n=0
        self.all_coef_values = pd.DataFrame(columns=self.column_names,index=range(len(self.x.columns)-1))
        for i in np.linspace(0.001,self.alpha_max,20):
            reg_model = self.technique(alpha=i).fit(self.x,self.y)
            model_scores = cross_validate(reg_model,self.x,self.y, 
                                          scoring=('r2', 'neg_mean_squared_error'),
                                          cv=self.kfolds,
                                          return_train_score=True)
            self.alphalist.append(i)
            self.r2valid_score.append(round(np.mean(model_scores['test_r2']),4))
            self.r2train_score.append(round(np.mean(model_scores['train_r2']),4))
            r2difference = abs(round(model_scores["train_r2"].mean(),4)-round(model_scores["test_r2"].mean(),4))
            self.r2difference.append(r2difference)
            self.msevalid_score.append(round(np.mean(model_scores['test_neg_mean_squared_error']),-4))
            self.msetrain_score.append(round(np.mean(model_scores['train_neg_mean_squared_error']),-4))
            msedifference = abs(round(model_scores["train_neg_mean_squared_error"].mean(),-4)-round(model_scores["test_neg_mean_squared_error"].mean(),-4))
            self.msedifference.append(msedifference)
            self.all_coef_values.loc[n] = reg_model.coef_
            n += 1
            self.non_zero_coefficients = reg_model.coef_[reg_model.coef_!=0]
            self.len_non_zero_coefs = len(self.non_zero_coefficients)
            self.len_all_coefs.append(self.len_non_zero_coefs)
            self.scores = pd.DataFrame({"Alpha":self.alphalist,
                                        "R^2 Train Score":self.r2train_score,"R^2 Validation Score":self.r2valid_score,
                                        "R^2 Difference":self.r2difference,"MSE Train Score":self.msetrain_score,
                                        "MSE Validation Score":self.msevalid_score,"MSE Difference":self.msedifference,
                                        "Number of Coefficients":self.len_all_coefs})
        self.all_coef_values.drop(["1"],axis=1,inplace=True)
        
        return self.scores
#===========================================================================================================================================================