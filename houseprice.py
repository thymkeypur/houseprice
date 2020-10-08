import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import median

# CHANGE FILE PATH
filepath = r'C:\Users\user\Desktop\train.csv'

house = pd.read_csv(filepath)

house.info()

for col in house.columns:
    print(house[col].value_counts())
    print('\n')

# Getting a list of columns with corr >= 0.4 against SalePrice  
corr = house.corr()['SalePrice'].apply(lambda x:abs(x))
corr_list = []
corr_col = []
count = 0
for c in corr:
    count = count + 1
    if c >= 0.4:
        corr_list.append(corr[[count-1]])
        corr_col.append(corr.index[count-1])   
corr_col
corr_list

# Quick plot to have a rough idea of the columns with corr >= 0.4 against SalePrice
for c in corr_col[:-1]:
    plt.title('Histogram for {}'.format(c))
    plt.hist(house[c])
    plt.show()
    
# Important variables for determining price: Year built, Year remodelled, Area/size & Overall quality
for c in corr_col[:-1]:
    print('Median pricing by: {}\n'.format(c))
    print(house.groupby(c)['SalePrice'].median())
    print('\n')
    
# Create a column total_built-in = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
# Take 1stFlrSF as the Land Size
house.loc[:,'BuiltIn'] = house.loc[:,'TotalBsmtSF'] + house.loc[:,'1stFlrSF'] + house.loc[:,'2ndFlrSF']
# Price per sf (land)
house['PriceSF(Land)'] = (house['SalePrice'] / house['1stFlrSF']).round(2)
# Price per sf (BuiltIn)
house['PriceSF(BuiltIn)'] = (house['SalePrice'] / house['BuiltIn']).round(2)

house = house.rename(columns={'1stFlrSF':'Land'})
  
# house1 focuses on columns with corr >= 0.4 against SalePrice
house1 = house[['OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','GrLivArea','FullBath','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageArea','Land','BuiltIn','PriceSF(Land)', 'PriceSF(BuiltIn)', 'SalePrice']]

# Get rid of outliers
for col in house1.columns:
    sns.boxplot(x=col, data=house1)
    plt.show()
    
Q1 = house1.quantile(0.25)
Q3 = house1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

house1.info()

# scrolls thru each column and check if between their upper & lower limit, else drop entry
def drop_outlier(df, col, upper, lower):
    keep_index = df[(df[col] <= upper) & (df[col] >= lower)].index    
    df = df.loc[keep_index,:]
    return df
    
for col in house1.columns:
    house1 = drop_outlier(house1, col, upper[col], lower[col])
    
# Create grouping for YearBuilt, YearRemodAdd, Land, BuiltIn for analysis purposes
    # Group Year by 10s
year_col = ['YearBuilt','YearRemodAdd','GarageYrBlt']
for col in year_col:
    house1[col+'_tens'] = house1[col].apply(lambda x:str(x)[:3]+'0s')
    
# Group Land and BuiltIn by: 500sf increments so if divided by 500
    # 0: < 500sf, 1: 500 to 999sf, 2: 1000 to 1499sf, 3: 1500 to 1999sf, 4: 2000 to 2499sf and so on...
house1['Land_grp'] = house1['Land']/500
house1['Land_grp'] = house1['Land_grp'].astype(int)
house1['Land_grp'].value_counts()

house1['BuiltIn_grp'] = house1['BuiltIn']/500
house1['BuiltIn_grp'] = house1['BuiltIn_grp'].astype(int)
house1['BuiltIn_grp'].value_counts()

# Plots for analysis
plot_col = ['YearBuilt_tens','YearRemodAdd_tens','GarageYrBlt_tens','Land_grp','BuiltIn_grp']
for col in plot_col:
    plt.figure(figsize=(15,8))
    plt.title('Sale Price based on {}'.format(col))
    sns.barplot(x=col, y='SalePrice', data=house1, estimator=median)
    plt.show()
    
plt.figure(figsize=(15,8))    
sns.barplot(x='Land_grp', y='SalePrice', data=house1, hue='Fireplaces', estimator=median)
plt.show()
     
# Train & test ML models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

models = []
models.append(('LR', LinearRegression()))
models.append(('RFR', RandomForestRegressor(n_estimators=100, max_depth=20)))
mae = []

for name, model in models:
    for seed in range(101):
        X_train, X_test, y_train, y_test = train_test_split(house1.drop(['SalePrice','YearBuilt_tens','YearRemodAdd_tens','GarageYrBlt_tens','Land_grp','BuiltIn_grp'],axis=1),house1['SalePrice'],test_size=0.3, random_state=100)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mae.append(mean_absolute_error(y_test, prediction))
        if (seed == 100):
            print(name, '\t:', np.mean(mae))
print('\nDone')
