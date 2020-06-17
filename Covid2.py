#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:14:26 2020

@author: moazam
"""

#Setup
import pandas as pd
import os

import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
import matplotlib as mpl
import matplotlib.pyplot as plt  

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "/Users/moazam/Documents"
CHAPTER_ID = "BigDataFinalProject"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR,  CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


#Load Data

FILE_PATH ='/Users/moazam/Documents/BigDataFinalProject'

def load_data(FILE_PATH=FILE_PATH):
    csv_path = os.path.join(FILE_PATH, "us-counties.csv")
    return pd.read_csv(csv_path)

covid = load_data()

#%matplotlib inline  -- Plot All Data in histogram

import matplotlib.pyplot as plt
covid.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()




#Convert Date to Epoch Date
covid.date = pd.to_datetime(covid.date)
covid.date = (covid.date - dt.datetime(1970,1,1)).dt.total_seconds()

#Plot all Data with time in hist
#%matplotlib inline   
import matplotlib.pyplot as plt
covid.hist(bins=50, figsize=(20,15))
save_fig("updated_attribute_histogram_plots")
plt.show()




#Convert Data Type: int to floats

convert_dict = {'cases': float, 'deaths': float } 
covid = covid.astype(convert_dict)


#--------------------------------------------------
#Create Test Set and Train Set

train_set, test_set = train_test_split(covid, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")
#--------------------------------------------------


#work on train set
covid = train_set.copy()

#Plot Train Set   ----> Helps to Discover and Visualize 

covid.plot(kind="scatter", x="cases", y="deaths", alpha=0.4,
    s=covid["cases"]/100, label="Cases", figsize=(10,7),
    c="deaths", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("covid_prices_scatterplot")



#Look for co-relations
corr_matrix = covid.corr()
corr_matrix["deaths"].sort_values(ascending=False)


#Different Attribute Combinations

#covid["cases_fips"] = covid["cases"]/covid["fips"]


# find co-relations with the new attributes
corr_matrix = covid.corr()
corr_matrix["deaths"].sort_values(ascending=False)

 # ----------Prepare data---------------

#Get rid of missing attributes
covid = covid.dropna(subset=["fips"])

#Drop Labels

covid_labels = covid["deaths"].copy()
covid = covid.drop("deaths", axis=1) 


#----------Data Cleaning--------------



covid_num = covid.drop(['county','state'], axis=1)


#Text Categories to Numbers
covid_cat = covid[['state']]

try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
    
    
    
ordinal_encoder = OrdinalEncoder()
covid_cat_encoded = ordinal_encoder.fit_transform(covid_cat)
covid_cat_encoded[:10]
    

ordinal_encoder.categories_



try:
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder 
    

cat_encoder = OneHotEncoder()
covid_cat_1hot = cat_encoder.fit_transform(covid_cat)
covid_cat_1hot

covid_cat_1hot.toarray()


cat_encoder = OneHotEncoder(sparse=False)
covid_cat_1hot = cat_encoder.fit_transform(covid_cat)
covid_cat_1hot

cat_encoder.categories_

#Transformation Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

covid_num_tr = num_pipeline.fit_transform(covid_num)




try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20m future_encoders import ColumnTransformer # Scikit-Learn < 0.20compose import ColumnTransformer
    
num_attribs = list(covid_num)
cat_attribs = ["state"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

covid_prepared = full_pipeline.fit_transform(covid)





#Model Selection
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(covid_prepared, covid_labels)




# let's try the full preprocessing pipeline on a few training instances
some_data = covid.iloc[:10]
some_labels = covid_labels.iloc[:10]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

def display_scores(scores):
    print("Linear Root Mean Square Error:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# RMSE
from sklearn.metrics import mean_squared_error

covid_predictions = lin_reg.predict(covid_prepared)
lin_mse = mean_squared_error(covid_labels, covid_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


from sklearn.model_selection import cross_val_score

lin_scores = cross_val_score(lin_reg, covid_prepared, covid_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)






lin_scores = cross_val_score(lin_reg, covid_prepared, covid_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# random Forest 
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(covid_prepared, covid_labels)


# forest_reg.score(covid_prepared, covid_labels)  

covid_predictions = forest_reg.predict(covid_prepared)
forest_mse = mean_squared_error(covid_labels, covid_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse



forest_scores = cross_val_score(forest_reg, covid_prepared, covid_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


#Fitting Real Data

scores = cross_val_score(lin_reg, covid_prepared, covid_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


#Grid Search


#Ful Pipeline

final_model = RandomForestRegressor(n_estimators=10, random_state=42)
final_model.fit(covid_prepared, covid_labels)
test_set = test_set.dropna(subset=["fips"])
X_test = test_set.drop("deaths", axis=1)
y_test = test_set["deaths"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(covid, covid_labels)
full_pipeline_with_predictor.predict(some_data)

some_labels

from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))




import csv
with open('prediction_output.csv', 'w',  newline  = '\n') as prediction_output:
    column_name = ['Prediction']
    thewriter = csv.DictWriter(prediction_output, fieldnames = column_name)
    thewriter.writeheader()

    for i in final_predictions:
        thewriter.writerow({'Prediction' : i })
    


with open('label_for_predictions.csv', 'w',  newline  = '\n') as output:
    column_name = ['Labels']
    thewriter = csv.DictWriter(output, fieldnames = column_name)
    thewriter.writeheader()

    for i in y_test:
        thewriter.writerow({'Labels' : i })