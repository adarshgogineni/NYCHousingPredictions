#!/usr/bin/env python
# coding: utf-8

# <!-- # <span style="color:#FFFFFF; font-size: 0%;">1 | Introduction</span>
# <div style="border-radius: 0px; background-color: #112241; text-align:center;">
#     <h4 style="color: white; padding: 1.5rem; font-size: 19px"><b>1 | INTRODUCTION</b></h4>
# </div>
# <!-- <br> -->
# 
# <div style="display: flex; flex-direction: row; align-items: center;">
#     <div style="flex: 0; margin-top: 8px;">
#         <img src="https://i.pinimg.com/474x/87/2f/38/872f388b608d7065788fc4ef3c8b83fb.jpg" alt="Image" style="max-width: 300px; max-height: 350px;" />
#     </div>
#     <div style="flex: 1; margin-left: 30px; margin-top: 6px">
#         <p style="font-weight: bold; color: black; font-size: 17px">Introduction</p>
#         <p>This notebook is created for New York Housing Market Data.
#         </p>
#         <p>This is a beginner-friendly notebook that attempts to perform Exploratory Data Analysis on the New York Housing Market Data and eventually train a <b>XGBoost</b> model on it and enhance the predictions by fine-tuning the model.
#         </p>
#         <p>Let's explore and then make results and discussion to gain deeper insights from our analysis. Let's explore and then make results and discussion to gain deeper insights from our analysis.</p>
#         <blockquote>  If you find this notebook helpful please consider upvoting ❤️</blockquote>
#     </div>
# </div>
# 
# 
# ## Contents:
# <hr>
# 
# 1. [Data Exploration](#data)
# 2. [Exploritory Data Analysis](#eda)
# 3. [Modeling](#model)
# 4. [Predictions](#hyper)
# 
# ### All the used libraries:
# 
# - Numpy
# - Pandas
# - Matplotlib
# - Seaborn
# - Folium
# - Scikit-learn
# - XGBoost
# - warnings
# 
# ### Models used to make predictions:
# 
# - XGBoost Classifier
# - GridSearchCV for Hyperparameter tuning
# 
# Now, let's import the data.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import folium
import numpy as np
from IPython.display import Markdown
from sklearn.preprocessing import LabelEncoder

def bold(string):
    display(Markdown(string))

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# In[2]:


rc = {
    "axes.facecolor": "#F8F8F8",
    "figure.facecolor": "#F8F8F8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4,
}

sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
mgt = Style.BRIGHT + Fore.MAGENTA
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL

plt.style.use('seaborn-v0_8-pastel')


# # 1. Data Exploration
# This dataset contains prices of New York houses, providing valuable insights into the real estate market in the region. It includes information such as broker titles, house types, prices, number of bedrooms and bathrooms, property square footage, addresses, state, administrative and local areas, street names, and geographical coordinates.
# 
# ## 1.1 Data Description
# 
# - **BROKERTITLE**: Title of the broker
# - **TYPE**: Type of the house
# - **PRICE**: Price of the house
# - **BEDS**: Number of bedrooms
# - **BATH**: Number of bathrooms
# - **PROPERTYSQFT**: Square footage of the property
# - **ADDRESS**: Full address of the house
# - **STATE**: State of the house
# - **MAIN_ADDRESS**: Main address information
# - **ADMINISTRATIVE_AREA_LEVEL_2**: Administrative area level 2 information
# - **LOCALITY**: Locality information
# - **SUBLOCALITY**: Sublocality information
# - **STREET_NAME**: Street name
# - **LONG_NAME**: Long name
# - **FORMATTED_ADDRESS**: Formatted address
# - **LATITUDE**: Latitude coordinate of the house
# - **LONGITUD**E: Longitude coordinate of the house
# 
# ## 1.2 Data

# In[3]:


df = pd.read_csv('/kaggle/input/new-york-housing-market/NY-House-Dataset.csv')
df.head()


# In[4]:


display(df.info())


# # 2. Exploratory Data Analysis
# 
# Exploratory Data Analysis (EDA) is an analysis approach that identifies general patterns in the data. These patterns include outliers and features of the data that might be unexpected. EDA is an important first step in any data analysis.
# 
# ## 2.1 Null Values:
# Missing data/Null values is defined as the values or data that is not stored (or not present) for some variable/s in the given dataset.Here is a list of popular strategies to handle missing values in a dataset
# 
# - Deleting the Missing Values
# - Imputing the Missing Values
# - Imputing the Missing Values for Categorical Features
# - Imputing the Missing Values using Sci-kit Learn Library
# - Using “Missingness” as a Feature
# 
# Let's see if our data has any missing values or not.

# In[5]:


sns.displot(data=df.isnull().melt(value_name='missing'),
    y='variable',
    hue='missing',
    multiple='fill',
    height=8,
#     width=10,
    aspect=1.6
)

# specifying a threshold value
plt.axvline(0.4, color='r')
plt.title('Null Values in Train Data', fontsize=13)
plt.show()


# ## 2.2 Top 500 Most and Least Costly Houses in New York

# In[6]:


data = df[['LONGITUDE', 'LATITUDE', 'PRICE', 'STREET_NAME']].copy()
data = data.sort_values(by=['PRICE'], ascending=False)
data  = data.head(500)

data.rename(columns = {'LONGITUDE':'lon', 'LATITUDE':'lat', 
                              'PRICE':'value', 'STREET_NAME':'name'}, inplace = True) 

m = folium.Map(location=[48, -102], tiles="OpenStreetMap", zoom_start=3)

for i in range(0,len(data)):
   folium.Marker(
      location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
      popup=data.iloc[i]['value'],
   ).add_to(m)

map_title = "500 Most Costly Houses"
title_html = f'<h4 style="position:absolute;z-index:10000;left:40vw" ><b>{map_title}</b></h4>'
m.get_root().html.add_child(folium.Element(title_html))

sw = data[['lat', 'lon']].min().values.tolist()
ne = data[['lat', 'lon']].max().values.tolist()

m.fit_bounds([sw, ne])

m


# In[7]:


data = df[['LONGITUDE', 'LATITUDE', 'PRICE', 'STREET_NAME']].copy()
data = data.sort_values(by=['PRICE'], ascending=True)
data  = data.head(500)

data.rename(columns = {'LONGITUDE':'lon', 'LATITUDE':'lat', 
                              'PRICE':'value', 'STREET_NAME':'name'}, inplace = True) 

m = folium.Map(location=[48, -102], tiles="OpenStreetMap", zoom_start=3)

for i in range(0,len(data)):
   folium.Marker(
      location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
      popup=data.iloc[i]['value'],
   ).add_to(m)

map_title = "500 Least Costly Houses"
title_html = f'<h4 style="position:absolute;z-index:10000;left:40vw" ><b>{map_title}</b></h4>'
m.get_root().html.add_child(folium.Element(title_html))

sw = data[['lat', 'lon']].min().values.tolist()
ne = data[['lat', 'lon']].max().values.tolist()

m.fit_bounds([sw, ne])

m


# ## 2.3 Data Cleaning
# 
# ### 1. Removing Unnecesory Columns

# In[8]:


df = df.drop(['LONGITUDE', 'LATITUDE', 'FORMATTED_ADDRESS', 'LONG_NAME', 'STREET_NAME', 'ADMINISTRATIVE_AREA_LEVEL_2', 'MAIN_ADDRESS', 'STATE', 'ADDRESS', 'BROKERTITLE'], axis=1)
df.head()


# ### 2. Renaming Columns

# In[9]:


df.rename(columns = {'PRICE':'price', 'BEDS':'beds', 
                     'BATH':'bath', 'PROPERTYSQFT':'area', 'LOCALITY': 'place', 
                     'SUBLOCALITY':'sublocality', 'TYPE': "type"}, inplace = True) 

df.head()


# ### 3. Removing some Outliers

# In[10]:


df = df.drop(df[df['price'] == 2147483647].index)
df = df.drop(df[df['price'] == 195000000].index)


# In[11]:


df.head()


# ## 2.3 Univariate Analysis
# 
# ### 1. Property Type:

# In[12]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='type', log=True)
plt.xticks(rotation=90)
plt.title('Distribution of property types')
plt.show()


# ### 2. Number of Beds:

# In[13]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='beds', log=True)
plt.title('Number of Beds')
plt.show()


# ### 3. Number of Baths:

# In[14]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='bath', log=True)
plt.xticks(rotation=90)
plt.title('Number of Baths')
plt.show()


# In[15]:


df['bath'] = df['bath'].apply(np.ceil)


# In[16]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='bath', log=True)
plt.xticks(rotation=90)
plt.title('Number of Baths')
plt.show()


# ### 4. Localities:

# In[17]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='place', log=True)
plt.xticks(rotation=90)
plt.title('Localities')
plt.show()


# ### 5. Sublocalities:

# In[18]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.countplot(data=df, x='sublocality', log=True)
plt.xticks(rotation=90)
plt.title('Sublocalities')
plt.show()


# ### 6. Property area in SquareFeet: 
# Distribution of various property's areas can be seen in following graph.

# In[19]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.histplot(data=df, x='area', bins=50, kde=True)
plt.ylim(0,500)
plt.ticklabel_format(style = 'plain')
fig.set(xlabel='')
plt.suptitle('Distribution of Area')
plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.histplot(data=df, x='area', bins=50, kde=True)
plt.ylim(0,1000)
plt.ticklabel_format(style = 'plain')
plt.show()


# ### 7. Price:
# Distribution of prices of houses can be seen in following graph.

# In[21]:


fig, ax = plt.subplots(figsize=(18, 4))
fig = sns.histplot(data=df, x='price', bins=50, kde=True)
plt.ylim(0,500)
plt.ticklabel_format(style = 'plain')
fig.set(xlabel='')
plt.suptitle('Distribution of Price')
plt.show()


# In[22]:


le = LabelEncoder()
df.place = le.fit_transform(df.place)
df.sublocality = le.fit_transform(df.sublocality)
df.type = le.fit_transform(df.type)


# ## 2.4 Bivariate Regression Analysis

# In[23]:


fig = plt.figure(figsize=(20, 10))

fig1 = fig.add_subplot(221); sns.regplot(data=df, x="area", y="price", x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig2 = fig.add_subplot(222); sns.regplot(data=df, x="beds", y="price", x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig3 = fig.add_subplot(223); sns.regplot(data=df, x="bath", y="price", x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig4 = fig.add_subplot(224); sns.regplot(data=df, x="type", y="price", x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

plt.suptitle('Regression Analysis with Price - First Order')
plt.tight_layout()
plt.show()

# -----------

fig = plt.figure(figsize=(20, 10))

fig1 = fig.add_subplot(221); sns.regplot(data=df, x="place", y="price", x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig2 = fig.add_subplot(222); sns.regplot(data=df, x="sublocality", y="price", x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

plt.tight_layout()
plt.show()


# In[24]:


fig = plt.figure(figsize=(20, 10))

fig1 = fig.add_subplot(221); sns.regplot(data=df, x="area", y="price", order=2 ,x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig2 = fig.add_subplot(222); sns.regplot(data=df, x="beds", y="price", order=2, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig3 = fig.add_subplot(223); sns.regplot(data=df, x="bath", y="price", order=2, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig4 = fig.add_subplot(224); sns.regplot(data=df, x="type", y="price", order=2, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

plt.suptitle('Regression Analysis with Price - Second Order')
plt.tight_layout()
plt.show()

# -----------

fig = plt.figure(figsize=(20, 10))

fig1 = fig.add_subplot(221); sns.regplot(data=df, x="place", y="price", order=2, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig2 = fig.add_subplot(222); sns.regplot(data=df, x="sublocality", y="price", order=2, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

plt.tight_layout()
plt.show()


# In[25]:


fig = plt.figure(figsize=(20, 10))

fig1 = fig.add_subplot(221); sns.regplot(data=df, x="area", y="price", logx=True ,x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig2 = fig.add_subplot(222); sns.regplot(data=df, x="beds", y="price", logx=True, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig3 = fig.add_subplot(223); sns.regplot(data=df, x="bath", y="price", logx=True, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig4 = fig.add_subplot(224); sns.regplot(data=df, x="type", y="price", logx=True, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

plt.suptitle('Regression Analysis with Price - Log Order')
plt.tight_layout()
plt.show()

# -----------

fig = plt.figure(figsize=(20, 10))

fig1 = fig.add_subplot(221); sns.regplot(data=df, x="place", y="price", logx=True, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

fig2 = fig.add_subplot(222); sns.regplot(data=df, x="sublocality", y="price", logx=True, x_jitter=.15, line_kws=dict(color="r"),)
plt.ticklabel_format(style = 'plain')

plt.tight_layout()
plt.show()


# # 3. Modeling
# 
# ## 3.1 Dropping unnecessary columns

# In[26]:


df = df.drop(['type', 'place', 'sublocality'], axis=1)
df.head()


# In[27]:


X = df.drop(['price'], axis=1)
y = df['price']


# In[28]:


X.head()


# In[29]:


y.head()


# ## 3.2 Encoding categorical variables

# In[30]:


X_encoded = pd.get_dummies(X, columns=['bath',
                                       'beds'])

X_encoded.head()


# ## 3.3 Splitting data into train and test

# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, 
                                                    y, 
                                                    random_state=42,)


# In[32]:


clf_xgb_v1 = xgb.XGBRegressor(objective="reg:squarederror",
                            # missing=None,
                            seed=42)

clf_xgb_v1.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=20,
            eval_metric='rmse',
            eval_set=[(X_test, y_test)])


# # 4. Predictions

# In[33]:


y_preds = clf_xgb_v1.predict(X_test)


# In[34]:


fig = plt.figure(figsize=(20, 10))
sns.regplot(data=df, x=y_test, y=y_preds, x_jitter=.15, line_kws=dict(color="r"),)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.ticklabel_format(style = 'plain')

