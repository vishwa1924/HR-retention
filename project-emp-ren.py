# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:48:28 2020

@author: raghuraj@hotmail.com


"""
##########################################################
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
color = sns.color_palette()
from IPython.display import display
pd.options.display.max_columns = None
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.set_config_file(offline=True)
import cufflinks
cufflinks.go_offline(connected=True)

##################################################

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer


# Read Excel file
df = pd.read_csv('C:/Users/USER/Desktop/EmployeeRetension.csv')
print("Shape of dataframe is: {}".format(df.shape))

# Make a copy of the original sourcefile
df_temp = df.copy()

print(df.info())              # Columns datatypes and missign values

def basic_info(df):
    d = {1:'one', 2:'two'}
    d['shape'] = df.shape
    d['columns'] = df.columns          # Dataset columns
    d['head'] = df.head()              # Dataset header
    # let's break down the columns by their type (i.e. int64, float64, object)
    d['datatypes'] = df.columns.to_series().groupby(df.dtypes).groups
    d['description'] = df.describe()   # Get details like count, mean, min, max etc.
    
    return(d)

basic_df_info = basic_info(df)

print(basic_df_info['shape'])  
print(basic_df_info['columns'])
print(basic_df_info['head'])
print(basic_df_info['datatypes'])
print(basic_df_info['description'])


import matplotlib.pyplot as plt
df.hist(figsize=(15,15))
plt.show()


from scipy.stats import norm #, skew

(mu, sigma) = norm.fit(df.loc[df['Attrition'] == 'Yes', 'Age'])
print('Ex-exmployees: average age = {:.1f} years old and standard deviation = {:.1f}'.format(mu, sigma))

(mu, sigma) = norm.fit(df.loc[df['Attrition'] == 'No', 'Age'])
print('Current exmployees: average age = {:.1f} years old and standard deviation = {:.1f}'.format(mu, sigma))

# Let's create a kernel density estimation (KDE) plot colored by the value of the target. 
# A kernel density estimation (KDE) is a non-parametric way to estimate the probability 
# density function of a random variable. It will allow us to identify if there is a 
# correlation between the Age of the Client and their ability to pay it back.

import seaborn as sns

plt.figure(figsize=(10,5))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'Age'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'Age'], label = 'Ex-Employees')
plt.xlim(left=18, right=60)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Age Distribution in Percent by Attrition Status');


# Education Field of employees
df['EducationField'].value_counts()    


df_EducationField = pd.DataFrame(columns=["Field", "% of Leavers"])
i=0
for field in list(df['EducationField'].unique()):
    numerator = df[(df['EducationField']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['EducationField']==field].shape[0]
    ratio =  numerator / denominator    # particular field / All fields
#    ratio = df[(df['EducationField']==field)&(df['Attrition']=="Yes")].shape[0] / df[df['EducationField']==field].shape[0]
    df_EducationField.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    

df_EF = df_EducationField.groupby(by="Field").sum()
df_EF.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_EF.iloc[:, 0],df_EF.iloc[:, 1])
plt.title('Leavers by Education Field (%)')
plt.show()


# Gender 
df['Gender'].value_counts()

print("Normalised gender distribution of ex-employees in the dataset: Male = {:.1f}%"
      .format((df[(df['Attrition'] == 'Yes') & (df['Gender'] == 'Male')].shape[0] / df[df['Gender'] == 'Male'].shape[0])*100))

print("Normalised gender distribution of ex-employees in the dataset: Female = {:.1f}%."
      .format((df[(df['Attrition'] == 'Yes') & (df['Gender'] == 'Female')].shape[0] / df[df['Gender'] == 'Female'].shape[0])*100))


df_Gender = pd.DataFrame(columns=["Gender", "% of Leavers"])
i=0
for field in list(df['Gender'].unique()):
    numerator = df[(df['Gender']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['Gender']==field].shape[0]
    ratio = numerator / denominator
    df_Gender.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_G = df_Gender.groupby(by="Gender").sum()
df_G.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_G.iloc[:, 0],df_G.iloc[:, 1])
plt.title('Leavers by Gender (%)')
plt.show()

# Marital Status
df['MaritalStatus'].value_counts()

df_Marital = pd.DataFrame(columns=["Marital Status", "% of Leavers"])
i=0
for field in list(df['MaritalStatus'].unique()):
    numerator = df[(df['MaritalStatus']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['MaritalStatus']==field].shape[0]
    ratio =  numerator / denominator
    df_Marital.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_MS = df_Marital.groupby(by="Marital Status").sum()
df_MS.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_MS.iloc[:, 0],df_MS.iloc[:, 1])
plt.title('Leavers by Marital Status (%)')
plt.show()


# Distance from home

print("Distance from home for employees to get to work is from {} to {} miles."
      .format(df['DistanceFromHome'].min(), df['DistanceFromHome'].max()))


print('Average distance from home for currently active employees: {:.2f} miles'
      .format(df[df['Attrition'] == 'No']['DistanceFromHome'].mean()))

print('Average distance from home for ex-employees: {:.2f} miles'
      .format(df[df['Attrition'] == 'Yes']['DistanceFromHome'].mean()))


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'DistanceFromHome'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'DistanceFromHome'], label = 'Ex-Employees')
plt.xlabel('DistanceFromHome')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Distance From Home in Percent by Attrition Status');


# Department

df['Department'].value_counts()


df_Department = pd.DataFrame(columns=["Department", "% of Leavers"])
i=0
for field in list(df['Department'].unique()):
    numerator = df[(df['Department']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['Department']==field].shape[0]
    ratio = numerator / denominator
    df_Department.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_DF = df_Department.groupby(by="Department").sum()
df_DF.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_DF.iloc[:, 0],df_DF.iloc[:, 1])
plt.title('Leavers by Department (%)')
plt.show()

# Business Travel and Role & work condition

df['BusinessTravel'].value_counts()

df_BusinessTravel = pd.DataFrame(columns=["Business Travel", "% of Leavers"])
i=0
for field in list(df['BusinessTravel'].unique()):
    numerator = df[(df['BusinessTravel']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['BusinessTravel']==field].shape[0]
    ratio = numerator / denominator
    df_BusinessTravel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_BT = df_BusinessTravel.groupby(by="Business Travel").sum()
df_BT.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_BT.iloc[:, 0],df_BT.iloc[:, 1])
plt.title('Leavers by Business Travels (%)')
plt.show()

# Employees in the database have several roles on-file
df['JobRole'].value_counts()

df_JobRole = pd.DataFrame(columns=["Job Role", "% of Leavers"])
i=0
for field in list(df['JobRole'].unique()):
    numerator = df[(df['JobRole']==field)&(df['Attrition']=="Yes")].shape[0] 
    denominator = df[df['JobRole']==field].shape[0] 
    ratio = numerator / denominator
    df_JobRole.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JR = df_JobRole.groupby(by="Job Role").sum()
df_JR.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_JR.iloc[:, 0],df_JR.iloc[:, 1])
plt.title('Leavers by Job Roles (%)')
plt.show()

# Job levels

df['JobLevel'].value_counts()

df_JobLevel = pd.DataFrame(columns=["Job Level", "% of Leavers"])
i=0
for field in list(df['JobLevel'].unique()):
    numerator = df[(df['JobLevel']==field)&(df['Attrition']=="Yes")].shape[0] 
    denominator = df[df['JobLevel']==field].shape[0]
    ratio = numerator / denominator
    df_JobLevel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JL = df_JobLevel.groupby(by="Job Level").sum()
df_JL.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_JL.iloc[:, 0],df_JL.iloc[:, 1])
plt.title('Leavers by Job Levels (%)')
plt.show()

# Job involvement

df['JobInvolvement'].value_counts()

df_JobInvolvement = pd.DataFrame(columns=["Job Involvement", "% of Leavers"])
i=0
for field in list(df['JobInvolvement'].unique()):
    numerator = df[(df['JobInvolvement']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['JobInvolvement']==field].shape[0]
    ratio = numerator / denominator
    df_JobInvolvement.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JI = df_JobInvolvement.groupby(by="Job Involvement").sum()
df_JI.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_JI.iloc[:, 0],df_JI.iloc[:, 1])
plt.title('Leavers by Job Involvment (%)')
plt.show()

# Employee  training times
print("Number of training times last year varies from {} to {} times."
      .format(df['TrainingTimesLastYear'].min(), df['TrainingTimesLastYear'].max()))


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'TrainingTimesLastYear'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'TrainingTimesLastYear'], label = 'Ex-Employees')
plt.xlabel('TrainingTimesLastYear')
plt.ylabel('Density')
plt.title('Training Times Last Year in Percent, by Attrition Status');


# Number of companies worked at
df['NumCompaniesWorked'].value_counts()

df_NumCompaniesWorked = pd.DataFrame(columns=["Num Companies Worked", "% of Leavers"])
i=0
for field in list(df['NumCompaniesWorked'].unique()):
    numerator = df[(df['NumCompaniesWorked']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['NumCompaniesWorked']==field].shape[0] 
    ratio = numerator / denominator
    df_NumCompaniesWorked.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_NC = df_NumCompaniesWorked.groupby(by="Num Companies Worked").sum()
df_NC.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_NC.iloc[:, 0],df_NC.iloc[:, 1])
plt.title('Leavers by number of companies worked (%)')
plt.show()

# Years at the company

print("Number of Years at the company varies from {} to {} years."
      .format(df['YearsAtCompany'].min(), df['YearsAtCompany'].max()))

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'YearsAtCompany'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'YearsAtCompany'], label = 'Ex-Employees')
plt.xlabel('YearsAtCompany')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years At Company in Percent by Attrition Status');

# Number of years at the current role
print("Number of Years in the current role varies from {} to {} years."
      .format(df['YearsInCurrentRole'].min(), df['YearsInCurrentRole'].max()))

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'YearsInCurrentRole'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'YearsInCurrentRole'], label = 'Ex-Employees')
plt.xlabel('YearsInCurrentRole')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years In Current Role in Percent by Attrition Status');

# Years since last promotion

print("Number of Years since last promotion varies from {} to {} years."
      .format(df['YearsSinceLastPromotion'].min(), df['YearsSinceLastPromotion'].max()))

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'YearsSinceLastPromotion'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'YearsSinceLastPromotion'], label = 'Ex-Employees')
plt.xlabel('YearsSinceLastPromotion')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years Since Last Promotion in Percent by Attrition Status');


# Total working years
print("Total working years varies from {} to {} years."
      .format(df['TotalWorkingYears'].min(), df['TotalWorkingYears'].max()))

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'TotalWorkingYears'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'TotalWorkingYears'], label = 'Ex-Employees')
plt.xlabel('TotalWorkingYears')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Total Working Years in Percent by Attrition Status');


# years with the current manager

print("Number of Years with the current manager varies from {} to {} years."
      .format(df['YearsWithCurrManager'].min(), df['YearsWithCurrManager'].max()))

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'YearsWithCurrManager'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'YearsWithCurrManager'], label = 'Ex-Employees')
plt.xlabel('YearsWithCurrManager')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years With the Current Manager in Percent by Attrition Status');

# Work-Life Balance Score

df['WorkLifeBalance'].value_counts()

df_WorkLifeBalance = pd.DataFrame(columns=["WorkLifeBalance", "% of Leavers"])
i=0
for field in list(df['WorkLifeBalance'].unique()):
    numerator = df[(df['WorkLifeBalance']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['WorkLifeBalance']==field].shape[0]
    ratio = numerator / denominator 
    df_WorkLifeBalance.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_WLB = df_WorkLifeBalance.groupby(by="WorkLifeBalance").sum()
df_WLB.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_WLB.iloc[:, 0],df_WLB.iloc[:, 1])
plt.title('Leavers by work life balance (%)')
plt.show()

# Standard hours
df['StandardHours'].value_counts()

# Overtime

df['OverTime'].value_counts()

df_OverTime = pd.DataFrame(columns=["OverTime", "% of Leavers"])
i=0
for field in list(df['OverTime'].unique()):
    numerator = df[(df['OverTime']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['OverTime']==field].shape[0]
    ratio = numerator / denominator
    df_OverTime.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_OT = df_OverTime.groupby(by="OverTime").sum()
df_OT.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_OT.iloc[:, 0],df_OT.iloc[:, 1])
plt.title('Leavers by Overtime (%)')
plt.show()

# Pay/Salary Employee Information

print("Employee Hourly Rate varies from ${} to ${}."
      .format(df['HourlyRate'].min(), df['HourlyRate'].max()))

print("Employee Daily Rate varies from ${} to ${}."
      .format(df['DailyRate'].min(), df['DailyRate'].max()))

print("Employee Monthly Rate varies from ${} to ${}."
      .format(df['MonthlyRate'].min(), df['MonthlyRate'].max()))

print("Employee Monthly Income varies from ${} to ${}."
      .format(df['MonthlyIncome'].min(), df['MonthlyIncome'].max()))

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'MonthlyIncome'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'MonthlyIncome'], label = 'Ex-Employees')
plt.xlabel('Monthly Income')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Monthly Income in Percent by Attrition Status');

# Percentage of salary hikes

print("Percentage Salary Hikes varies from {}% to {}%."
      .format(df['PercentSalaryHike'].min(), df['PercentSalaryHike'].max()))


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'PercentSalaryHike'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'PercentSalaryHike'], label = 'Ex-Employees')
plt.xlabel('PercentSalaryHike')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Percent Salary Hike in Percent by Attrition Status');


# Stock options
print("Stock Option Levels varies from {} to {}."
      .format(df['StockOptionLevel'].min(), df['StockOptionLevel'].max()))

print("Normalised percentage of leavers by Stock Option Level 1: {:.2f}%"
      .format(df[(df['Attrition'] == 'Yes') & (df['StockOptionLevel'] == 1)].shape[0] 
              / df[df['StockOptionLevel'] == 1].shape[0]*100))

print("Normalised percentage of leavers by Stock Option Level 2: {:.2f}%"
      .format(df[(df['Attrition'] == 'Yes') & (df['StockOptionLevel'] == 2)].shape[0] 
              / df[df['StockOptionLevel'] == 1].shape[0]*100))

print("Normalised percentage of leavers by Stock Option Level 3: {:.2f}%"
      .format(df[(df['Attrition'] == 'Yes') & (df['StockOptionLevel'] == 3)].shape[0] 
              / df[df['StockOptionLevel'] == 1].shape[0]*100))


df_StockOptionLevel = pd.DataFrame(columns=["StockOptionLevel", "% of Leavers"])
i=0
for field in list(df['StockOptionLevel'].unique()):
    ratio = df[(df['StockOptionLevel']==field)&(df['Attrition']=="Yes")].shape[0] / df[df['StockOptionLevel']==field].shape[0]
    df_StockOptionLevel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_SOL = df_StockOptionLevel.groupby(by="StockOptionLevel").sum()
df_SOL.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_SOL.iloc[:, 0],df_SOL.iloc[:, 1])
plt.title('Leavers by Stock Option Level (%)')
plt.show()


# Employee Satisfaction and Performance Information

# EnvironmentSatisfaction

df['EnvironmentSatisfaction'].value_counts()


df_EnvironmentSatisfaction = pd.DataFrame(columns=["EnvironmentSatisfaction", "% of Leavers"])
i=0
for field in list(df['EnvironmentSatisfaction'].unique()):
    numerator = df[(df['EnvironmentSatisfaction']==field)&(df['Attrition']=="Yes")].shape[0] 
    denominator = df[df['EnvironmentSatisfaction']==field].shape[0] 
    ratio = numerator / denominator
    df_EnvironmentSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_ENV = df_EnvironmentSatisfaction.groupby(by="EnvironmentSatisfaction").sum()
df_ENV.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_ENV.iloc[:, 0],df_ENV.iloc[:, 1])
plt.title('Leavers by Environment Satisfaction (%)')
plt.show()

# Job satisfaction

df['JobSatisfaction'].value_counts()

df_JobSatisfaction = pd.DataFrame(columns=["JobSatisfaction", "% of Leavers"])
i=0
for field in list(df['JobSatisfaction'].unique()):
    numerator = df[(df['JobSatisfaction']==field)&(df['Attrition']=="Yes")].shape[0] 
    denominator = df[df['JobSatisfaction']==field].shape[0]
    ratio = numerator / denominator
    df_JobSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JS = df_JobSatisfaction.groupby(by="JobSatisfaction").sum()
df_JS.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_JS.iloc[:, 0],df_JS.iloc[:, 1])
plt.title('Leavers by Job satisfaction (%)')
plt.show()

# RelationshipSatisfaction

df['RelationshipSatisfaction'].value_counts()

df_RelationshipSatisfaction = pd.DataFrame(columns=["RelationshipSatisfaction", "% of Leavers"])
i=0
for field in list(df['RelationshipSatisfaction'].unique()):
    numerator = df[(df['RelationshipSatisfaction']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['RelationshipSatisfaction']==field].shape[0]
    ratio = numerator / denominator
    df_RelationshipSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_RS = df_RelationshipSatisfaction.groupby(by="RelationshipSatisfaction").sum()
df_RS.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_RS.iloc[:, 0],df_RS.iloc[:, 1])
plt.title('Leavers by Job satisfaction (%)')
plt.show()

# Employee Performance Rating

df['PerformanceRating'].value_counts()

print("Normalised percentage of leavers by Stock Option Level 3: {:.2f}%"
      .format(df[(df['Attrition'] == 'Yes') & (df['PerformanceRating'] == 3)].shape[0]
              / df[df['StockOptionLevel'] == 1].shape[0]*100))

print("Normalised percentage of leavers by Stock Option 4: {:.2f}%"
      .format(df[(df['Attrition'] == 'Yes') & (df['PerformanceRating'] == 4)].shape[0] 
              / df[df['StockOptionLevel'] == 1].shape[0]*100))

df_PerformanceRating = pd.DataFrame(columns=["PerformanceRating", "% of Leavers"])
i=0
for field in list(df['PerformanceRating'].unique()):
    numerator = df[(df['PerformanceRating']==field)&(df['Attrition']=="Yes")].shape[0]
    denominator = df[df['PerformanceRating']==field].shape[0]
    ratio = numerator / denominator
    df_PerformanceRating.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_PR = df_PerformanceRating.groupby(by="PerformanceRating").sum()
df_PR.reset_index(inplace=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df_PR.iloc[:, 0],df_PR.iloc[:, 1])
plt.title('Leavers by Job satisfaction (%)')
plt.show()

# Attrition

df['Attrition'].value_counts()


print("Percentage of Current Employees is {:.1f}% "
      .format(df[df['Attrition'] == 'No'].shape[0] / df.shape[0]*100))

print("Percentage of Ex-employees is {:.1f}%"
      .format(df[df['Attrition'] == 'Yes'].shape[0] / df.shape[0]*100))

# Correlation

# Find correlations with the target and sort
df_trans = df.copy()
df_trans['Target'] = df_trans['Attrition'].apply(lambda x: 0 if x == 'No' else 1)
df_trans = df_trans.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)
correlations = df_trans.corr()['Target'].sort_values()

print('Most Positive Correlations: \n', correlations.tail(5))
print('\nMost Negative Correlations: \n', correlations.head(5))

# Calculate correlations

corr = df_trans.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr, vmax=.5, mask=mask,
            # annot=True, fmt='.2f',
            linewidths=.2, cmap="YlGnBu")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()

print(df.shape)
df.head()

# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in df.columns[1:]:
    if df[col].dtype == 'object':
        if len(list(df[col].unique())) <= 2:
            print(col)
            le.fit(df[col])
            df[col] = le.transform(df[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))

# convert rest of categorical variable into dummy
df = pd.get_dummies(df, drop_first=True)

print(df.shape)
df.head()

'''
Feature Scaling
Feature Scaling using MinMaxScaler essentially shrinks the range such that the 
range is now between 0 and n. Machine Learning algorithms perform better when 
input numerical variables fall within a similar scale. In this case, we are 
scaling between 0 and 5.
'''

# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
HR_col = list(df.columns)
HR_col.remove('Attrition')
for col in HR_col:
    df[col] = df[col].astype(float)
    df[[col]] = scaler.fit_transform(df[[col]])
df['Attrition'] = pd.to_numeric(df['Attrition'], downcast='float')
df.head()

print('Size of Full Encoded Dataset: {}'. format(df.shape))

# Splitting data into training and testing sets

# assign the target to a new dataframe and convert it to a numerical feature
target = df['Attrition'].copy()

# let's remove the target feature and redundant features from the dataset
df.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace=True)
print('Size of Full dataset is: {}'.format(df.shape))

# Since we have class imbalance (i.e. more employees with Attrition=0 than Attrition=1)
# let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=7, stratify=target)  
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# Building Machine Learning Models

# The algorithms considered in this section are: 
# Logistic Regression, Random Forest, SVM, KNN, Decision Tree Classifier, Gaussian NB.


# selection of algorithms to consider and set performance measure
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')))
models.append(('Random Forest', RandomForestClassifier(n_estimators=100, random_state=7)))
models.append(('SVM', SVC(gamma='auto', random_state=7)))
models.append(('KNN', KNeighborsClassifier())) 
models.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=7)))
models.append(('Gaussian NB', GaussianNB()))


# Let's evaluate each model in turn and provide accuracy and standard deviation scores

############################################################################

# Explaining accuracy_score

import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3, 4]
y_true = [0, 1, 2, 3, 4]
accuracy_score(y_true, y_pred)    # Gives a fraction value
accuracy_score(y_true, y_pred, normalize=False)    # Gives a number

# Explaining roc_auc_score

import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_pred)


###########################################################################

acc_results = []
auc_results = []
names = []

# set table to populate with performance results
col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD', 'Accuracy Mean', 'Accuracy STD']
df_results = pd.DataFrame(columns=col)
i = 0
# evaluate each model using cross-validation
kfold = model_selection.KFold(n_splits=10, random_state=7)  # 10-fold cross-validation

for name, model in models:

    cv_acc_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy') # accuracy scoring

    cv_auc_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')   # roc_auc scoring

    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    names.append(name)
    df_results.loc[i] = [name,
                         round(cv_auc_results.mean()*100, 2),
                         round(cv_auc_results.std()*100, 2),
                         round(cv_acc_results.mean()*100, 2),
                         round(cv_acc_results.std()*100, 2)
                         ]
    i += 1
df_results.sort_values(by=['ROC AUC Mean'], ascending=False)

# Box plot using accuracy score

fig = plt.figure(figsize=(10, 6))
fig.suptitle('Algorithm Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(acc_results)
ax.set_xticklabels(names)
plt.show()

# Box plot using roc auc score

fig = plt.figure(figsize=(10, 6))
fig.suptitle('Algorithm ROC AUC Comparison')
ax = fig.add_subplot(111)
plt.boxplot(auc_results)
ax.set_xticklabels(names)
plt.show()


'''
Logistic Regression
Let's take a closer look at using the Logistic Regression algorithm. 
We will be using 10 fold Cross-Validation to train our Logistic Regression Model and estimate its AUC score.

'''

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression(solver='liblinear', class_weight="balanced", random_state=7)
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))

'''
Fine-tuning
GridSearchCV allows use to fine-tune hyper-parameters by searching specified parameter values for 
an estimator.
'''

param_grid = {'C': np.arange(1e-03, 2, 0.01)} # hyper-parameter list to fine-tune
log_gs = GridSearchCV(LogisticRegression(solver='liblinear', # setting GridSearchCV
                                         class_weight="balanced", 
                                         random_state=7),
                      iid=True,
                      return_train_score=True,
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=10)

log_grid = log_gs.fit(X_train, y_train)
log_opt = log_grid.best_estimator_
results = log_gs.cv_results_

print('='*50)
print("best params: " + str(log_gs.best_estimator_))
print("best params: " + str(log_gs.best_params_))
print('best score:', log_gs.best_score_)
print('='*50)


# Evaluation
## Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, log_opt.predict(X_test))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print('Accuracy of Logistic Regression Classifier on test set: {:.2f}'.format(log_opt.score(X_test, y_test)*100))


# Calculating probabilities in Logistic Regression 

log_opt.fit(X_train, y_train) # fit optimised model to the training data
probs = log_opt.predict_proba(X_test) # predict probabilities
probs = probs[:, 1] # we will only keep probabilities associated with the employee leaving
logit_roc_auc = roc_auc_score(y_test, probs) # calculate AUC score using test dataset

print('AUC score: %.3f' % logit_roc_auc)

# Random Forest Classifier
# Fine-tuning

rf_classifier = RandomForestClassifier(class_weight = "balanced", random_state=7)
#param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175],
#              'min_samples_split':[2,4,6,8,10],
#              'min_samples_leaf': [1, 2, 3, 4],
#              'max_depth': [5, 10, 15, 20, 25]}

param_grid = {'n_estimators': [50, 75],
              'min_samples_split':[2,6,10],
              'min_samples_leaf': [2, 3, 4],
              'max_depth': [5, 10,  20]}

grid_obj = GridSearchCV(rf_classifier,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=10)

grid_fit = grid_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_

print('='*50)
print("best params: " + str(grid_obj.best_estimator_))
print("best params: " + str(grid_obj.best_params_))
print('best score:', grid_obj.best_score_)
print('='*50)

# features by their importance.

importances = rf_opt.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
names = [X_train.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Feature Importance") # Create plot title
plt.bar(range(X_train.shape[1]), importances[indices]) # Add bars
plt.xticks(range(X_train.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.show() # Show plot

# Random Forest helps us identify the 10 most important indicators (ranked in the table below).

importances = rf_opt.feature_importances_
df_param_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])
for i in range(44):
    feat = X_train.columns[i]
    coeff = importances[i]
    df_param_coeff.loc[i] = (feat, coeff)
df_param_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)
df_param_coeff = df_param_coeff.reset_index(drop=True)
df_param_coeff.head(10)

# Evaluation
## Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, rf_opt.predict(X_test))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print('Accuracy of RandomForest Regression Classifier on test set: {:.2f}'.format(rf_opt.score(X_test, y_test)*100))

# Calculating the probabilities 

rf_opt.fit(X_train, y_train) # fit optimised model to the training data
probs = rf_opt.predict_proba(X_test) # predict probabilities
probs = probs[:, 1] # we will only keep probabilities associated with the employee leaving
rf_opt_roc_auc = roc_auc_score(y_test, probs) # calculate AUC score using test dataset
print('AUC score: %.3f' % rf_opt_roc_auc)


# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, log_opt.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_opt.predict_proba(X_test)[:,1])
plt.figure(figsize=(14, 6))

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_opt_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate ' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# Checking:

import random

# Get 3 random numbers in the range of X_test rows  
nums = random.sample(range(0, len(X_test)-1),  3)    

# Get 3 random rows from testing dataset
X_check = X_test.iloc[nums,]

# Predict the probabilities of the 3 employees  
check_probs = log_opt.predict_proba(X_check) 

# Save probabilities of only "Exit"inf prople
check_probs = check_probs[:, 1]

# Find the risk associated with the 3 employees
for i in range(len(check_probs)):
    if check_probs[i] < 0.6:
        print(" Employee number: ", i, " is in Low-risk zone")
    elif check_probs[i] > 0.6 and check_probs[i] <= 0.8:
        print(" Employee number: ", i, " is in Medium-risk zone")
    else:
        print(" Employee number: ", i, " is in High-risk zone")
        
        
        


