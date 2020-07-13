#!/usr/bin/env python
# coding: utf-8

# ## Retail Assessment :  
# #### Dated: 12.July.2020
# #### Author: Raghav Arora

# #### Objective: <br> <ol><li> Select the best performing products as to optimise the storage space.</li><br><li> Predict the demand for products as to ensure fast delivery in future</li></ol>

# ### Installing necessary python modules

# In[2]:


import pandas as pd, datetime,  numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


# ### Importing the dataset

# In[3]:


data =  pd.read_csv('data.csv', encoding = "ISO-8859-1", low_memory  = False)


# ## 1. Data Exploration and Cleaning

# ### Previewing the dataset

# In[4]:


data.head()


# In[5]:


data.describe()


# #### Findings: Quantity and Unit Price has negative values, as per my understanding of the retail dataset, these cases can be assumed as Returns.

# ### Outliers Detection

# In[6]:


import seaborn as sns
sns.boxplot(x=data['Quantity'])


# In[7]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(data['Quantity']))
print(z)
threshold = 5
print(np.where(z > threshold))


# ### Checking missing values

# In[8]:


data.isna().sum()


# ### Quantum of missing values in %

# In[9]:


print('Number of rows with blank Customer IDs:',round((data['CustomerID'].isna().sum() * 100)/len(data),2))
print('Number of rows with Description:',round((data['Description'].isna().sum() * 100)/len(data),2))


# #### Countries

# In[10]:


data['Country'].unique()


# #### Cases with Unit price <=0

# In[11]:


((data['UnitPrice'] <= 0).sum()*100)/len(data)


# #### Cases with Quantity <=0

# In[12]:


((data['Quantity'] <= 0).sum()*100)/len(data)


# #### Cases with Quantity <=0 OR Unit price <=0

# In[13]:


(((data['Quantity'] <= 0) | (data['UnitPrice'] <= 0)).sum()*100)/len(data)


# #### Cases with Quantity <=0 OR Unit price <=0 OR Quantity against each stock in an invoice >=2000

# In[14]:


(((data['Quantity'] <= 0) | (data['UnitPrice'] <= 0) | (data['Quantity'] > 2000)).sum()*100)/len(data)


# #### Cases with Quantity <=0 OR Unit Price <=0 OR Blank Customer ID

# In[15]:


(((data['Quantity'] <= 0) | (data['UnitPrice'] <= 0) | (data['CustomerID'].isna() ==  True)).sum()*100)/len(data)


# ### Cleaning the data

# ### The dataset contains rows where the Quantity < 0, (these are either discounts or returns).

# #### Also, unit price has to be greater than 0, in the case of purchase

# In[16]:


data_purchased=data.loc[(data['UnitPrice']>0) & (data['Quantity']>0) & (data['Quantity']<=2000)]


# In[17]:


len(data_purchased)


# In[18]:


data_purchased['InvoiceDate'] = pd.to_datetime(data_purchased['InvoiceDate'])


# In[19]:


data_purchased.head()


# In[20]:


len(data_purchased.drop_duplicates())


# In[21]:


len(data_purchased[['InvoiceNo', 'StockCode' ]].drop_duplicates())


# #### Calculating Total Amount against Stock in an invoice

# In[22]:


data_purchased['Amount'] = data_purchased['Quantity'] * data_purchased['UnitPrice']


# ## 2.  Select the best performing products as to optimise the storage space.

# #### Ranking the Stock based on the sold amount

# In[23]:


data_purchased_stockwise_amount=data_purchased.groupby(['StockCode','Description']).agg({'Amount':sum}).sort_values(by='Amount', ascending =False)


# #### Top n Stocks

# In[24]:


n=5


# In[25]:


data_purchased_stockwise_amount.head(n)


# #### Exporting the data

# In[26]:


data_purchased_stockwise_amount.to_csv('outputs/data_purchased_stockwise_amount.csv')


# #### Top Stocks on the basis of Sold Amount (Country wise)

# In[27]:


data_purchased_stockwise_amount_country_wise=data_purchased.groupby(['Country','StockCode','Description']).agg({'Amount':sum}).sort_values(by='Amount', ascending =False)
# 


# In[28]:


data_purchased_stockwise_amount_country_wise['Rank']=data_purchased_stockwise_amount_country_wise.groupby('Country').rank(ascending=False)


# #### Top n stocks by country

# In[29]:


data_purchased_stockwise_amount_country_wise.loc[data_purchased_stockwise_amount_country_wise["Rank"]<=n]


# ##### Exporting the data

# In[30]:


data_purchased_stockwise_amount_country_wise.to_csv('outputs/data_purchased_stockwise_amount_country_wise.csv')


# ## 3.  Predict the demand for products as to ensure fast delivery in future

# ### Preparing time series data for forecasting the demand

# ##### Extracting date from datetime

# In[31]:


data_purchased['Date']=data_purchased['InvoiceDate'].dt.date


# ##### Casting date as Year-Month

# In[32]:


data_purchased['Year_month']=data_purchased['InvoiceDate'].dt.strftime('%Y%m')


# #### Creating timeseries data ( 'Stock-Decsription', 'Year-month') wise sold quantity

# In[33]:


data_purchased_stock_ts=data_purchased.groupby(['Description', 'Year_month']).agg({'Quantity':sum}).reset_index()


# In[34]:


data_purchased_stock_ts_reshaped=data_purchased_stock_ts.loc[data_purchased_stock_ts['Quantity'] > 6].pivot_table(index='Description',columns='Year_month',values='Quantity', aggfunc=sum)


# In[35]:


data_purchased_stock_ts_reshaped


# In[36]:


data_purchased_stock_ts_reshaped['Not Blank']=data_purchased_stock_ts_reshaped.notnull().sum(axis=1)


# In[37]:


data_purchased_stock_ts_reshaped.head(5)


# In[38]:


data_purchased_stock_ts_reshaped.loc[data_purchased_stock_ts_reshaped['Not Blank']>6]


# #### Declaring the output dataframe

# In[39]:


pred_stock =pd.DataFrame()
pred_stock.index.name = 'Description'
pred_stock['Tplusone']=''
pred_stock['Tplustwo']=''
pred_stock['Tplusthree']=''


# ### Forecasting demand
# <br>Points to be considered:
# <ol><li>We have considered Exponential Smoothing for forecasting demand</li>
#     <li>The smoothing level (Alpha) and smoothing slope (beta) are taken as default (alpha = 0.8, beta = 0.2)</li>
#     <li>The stock which sold for more than 6 months were considered.</li></ol>

# In[40]:


for i in data_purchased_stock_ts_reshaped.loc[data_purchased_stock_ts_reshaped['Not Blank']>6].index.unique():
    
    X = data_purchased_stock_ts[['Year_month','Quantity']].loc[data_purchased_stock_ts['Description']==i].set_index(
'Year_month')
    fit2 = Holt(np.asarray(X), exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.3, optimized=False)
    predict=fit2.forecast(3)
    A=list(predict)
    pred_stock.loc[i]=A


# #### Predicted quantities (Stock Description wise) : for 2012 - Jan, Feb, Mar

# In[41]:


pred_stock.head(10)


# #### Exporting the result

# In[42]:


pred_stock.to_csv('outputs/Stock_Wise_Predicted_quant.csv')


# ## End of the Assessment
