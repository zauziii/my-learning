# Import Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
sns.set_style("whitegrid")
sns.set_style({'font.sans-serif':['simhei','Arial']})
%matplotlib inline

# Version Check 
from sys import version_info
if version_info.major != 3:
    raise Exception('Please use Python 3!')
# We used Python 3.11.9

# Import data (lianjia.csv)
lianjia_df = pd.read_csv('lianjia.csv')

# Show the first 6-row of the data
display(lianjia_df.head(6))
# There are 12 columns in the data. 
# Price is the target variable and the others are explanatory variabels.

# Check missing values
lianjia_df.info()
# There are 23677 rows in the data.
# It is obvious that there are misisng values in "Elevator".

# Summary statistics
lianjia_df.describe()
# From above, we can find that the minimum size of size is 2, which is not reasonable.

# Add a new feature: the average price
df = lianjia_df.copy()
df['AvgPrice'] = df['Price']/df['Size']

# Let's rearrange the columns
columns = ['Region', 'District', 'Garden', 'Layout', 
          'Floor', 'Year', 'Size', 'Elevator', 
          'Direction', 'Renovation', 'AvgPrice', 'Price']
df = pd.DataFrame(df, columns=columns)

# Check again
display(lianjia_df.head(6))
# Id is of no use in the analysis, remove it!

# EDA
# Data visualization

# Region
# Group by region and compare the price and number 
df_house_count = df.groupby('Region')['Price'].count().sort_values(ascending=False).to_frame().reset_index()
# Group by region and compare the average price and number 
df_house_mean = df.groupby('Region')['AvgPrice'].mean().sort_values(ascending=False).to_frame().reset_index()

f, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=(20,15))
sns.barplot(x='Region', y='AvgPrice', palette="Blues_d", data=df_house_mean, ax=ax1)
ax1.set_title('Comparison of unit price per square meter of second-hand housing in Beijing',fontsize=15)
ax1.set_xlabel('Region')
ax1.set_ylabel('Price Per Square Meter')
sns.barplot(x='Region', y='Price', palette="Greens_d", data=df_house_count, ax=ax2)
ax2.set_title('Comparison of the number of second-hand houses in Beijing',fontsize=15)
ax2.set_xlabel('Region')
ax2.set_ylabel('Number')
sns.boxplot(x='Region', y='Price', data=df, ax=ax3)
ax3.set_title('The total price of second-hand houses in Beijing',fontsize=15)
ax3.set_xlabel('Region')
ax3.set_ylabel('Total Price')
plt.show()

# Size
f, [ax1,ax2] = plt.subplots(1, 2, figsize=(15, 5))
# The distribution of building time
sns.histplot(x = 'Size', kde = True,
             line_kws = {'linestyle':'dashed',
                         'linewidth':'2'}, data=df, ax=ax1).lines[0].set_color('red')
# The relationship between building time and selling price
sns.regplot(x='Size', y='Price', data=df, ax=ax2)
plt.show()
# Size is long-tailed distributed, which indicates that there are many second-hand houses with large areas and beyond the normal range.
# Size and Price are linear related - The larger the area, the higher the price. However, outliers exist.
# 1. The area is less than 10 square meters, but the price exceeds 100 million yuan; 2. An area of more than 1000 square meters, the price is very low, need to check what is the situation.
df.loc[df['Size']<10]
df.loc[df['Size']>1000]
# Remove outliers and do the visualization again
df = df[(df['Layout']!='叠拼别墅')&(df['Size']<1000)]
f, [ax1,ax2] = plt.subplots(1, 2, figsize=(15, 5))
# The distribution of building time
sns.histplot(x = 'Size', kde = True,
             line_kws = {'linestyle':'dashed',
                         'linewidth':'2'}, data=df, ax=ax1).lines[0].set_color('red')
# The relationship between building time and selling price
sns.regplot(x='Size', y='Price', data=df, ax=ax2)
plt.show()
# No outliers detected!

# Layout
f, ax1= plt.subplots(figsize=(20,20))
sns.countplot(y='Layout', data=df, ax=ax1)
ax1.set_title('Housing type',fontsize=15)
ax1.set_xlabel('Number')
ax1.set_ylabel('Housing type')
plt.show()

# Renovation
df['Renovation'].value_counts()
df['Renovation'] = df.loc[(df['Renovation'] != '南北'), 'Renovation']
f, [ax1,ax2,ax3] = plt.subplots(1, 3, figsize=(20, 5))
sns.countplot(x ='Renovation', data=df, ax=ax1)
sns.barplot(x='Renovation', y='Price', data=df, ax=ax2)
sns.boxplot(x='Renovation', y='Price', data=df, ax=ax3)
plt.show()

# Elevator
misn = len(df.loc[(df['Elevator'].isnull()), 'Elevator'])
print('Number of missing value：'+ str(misn))
df['Elevator'] = df.loc[(df['Elevator'] == '有电梯')|(df['Elevator'] == '无电梯'), 'Elevator']
df.loc[(df['Floor']>6)&(df['Elevator'].isnull()), 'Elevator'] = '有电梯'
df.loc[(df['Floor']<=6)&(df['Elevator'].isnull()), 'Elevator'] = '无电梯'
f, [ax1,ax2] = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(x='Elevator', ax=ax1, data=df)
ax1.set_title('有无电梯数量对比',fontsize=15)
ax1.set_xlabel('是否有电梯')
ax1.set_ylabel('数量')
sns.barplot(x='Elevator', y='Price', data=df, ax=ax2)
ax2.set_title('有无电梯房价对比',fontsize=15)
ax2.set_xlabel('是否有电梯')
ax2.set_ylabel('总价')
plt.show()

# Year
grid = sns.FacetGrid(df, row='Elevator', col='Renovation', palette='seismic', height=5, aspect=1)
grid.map(plt.scatter, 'Year', 'Price')
grid.add_legend()

# Floor
f, ax1= plt.subplots(figsize=(20,5))
sns.countplot(x='Floor', data=df, ax=ax1)
ax1.set_title('Housing Type',fontsize=15)
ax1.set_xlabel('Number')
ax1.set_ylabel('Type')

# Feature engineering
# Outliers in Layout and Size
df = df[(df['Layout']!='叠拼别墅')&(df['Size']<1000)]
# Remove error rows in Renovation
df['Renovation'] = df.loc[(df['Renovation'] != '南北'), 'Renovation']
# For Elevator
df['Elevator'] = df.loc[(df['Elevator'] == '有电梯')|(df['Elevator'] == '无电梯'), 'Elevator']
# Impute missing value in Elevator
df.loc[(df['Floor']>6)&(df['Elevator'].isnull()), 'Elevator'] = '有电梯'
df.loc[(df['Floor']<=6)&(df['Elevator'].isnull()), 'Elevator'] = '无电梯'
# New feature
df['Layout_room_num'] = df['Layout'].str.extract('(^\d).*', expand=False).astype('int64')
df['Layout_hall_num'] = df['Layout'].str.extract('^\d.*?(\d).*', expand=False).astype('int64')
# For Year
df['Year'] = pd.qcut(df['Year'],8).astype('object')
# For direction
d_list_one = ['东','西','南','北']
d_list_two = ['东西','东南','东北','西南','西北','南北']
d_list_three = ['东西南','东西北','东南北','西南北']
d_list_four = ['东西南北']
def direct_func(x):
    a = list(set(x))
    b = ''.join(a)
    return(b)
df['Direction'] = df['Direction'].apply(direct_func)
df = df.loc[(df['Direction']!='no')&(df['Direction']!='nan')]
# Create new feature
df['Layout_total_num'] = df['Layout_room_num'] + df['Layout_hall_num']
df['Size_room_ratio'] = df['Size']/df['Layout_total_num']
# Delete unused features
df = df.drop(['Layout','AvgPrice','Garden', 'District', 'Id'],axis=1)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns = ['Region','Year','Elevator','Direction','Renovation'], dtype=int)

# data_corr 
colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_encoded.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

# Modelling
features = np.array(df_encoded.loc[:, df_encoded.columns != 'Price'])
prices = np.array(df_encoded['Price'])

# train and split
from sklearn.model_selection import train_test_split
features_train, features_test, prices_train, prices_test = train_test_split(features, prices, test_size=0.2, random_state=0)

# Import modules for modelling
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Helper functions
def fit_model(X, y):
    cross_validator = KFold(10, shuffle=True)
    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(estimator = regressor, param_grid = params, scoring = scoring_fnc, cv = cross_validator)
    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(X, y)
    # print pd.DataFrame(grid.cv_results_)
    return grid.best_estimator_

def performance_metric(y_true, y_predict):
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    return score

