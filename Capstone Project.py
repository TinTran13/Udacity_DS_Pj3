# %% [markdown]
# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This dataset simulates customer behavior on the Starbucks rewards mobile app, where users receive different types of offers. Your task is to analyze transaction, demographic, and offer data to determine which demographic groups respond best to which offer type. The data represents a simplified version of the Starbucks app, and it includes information about offers' validity periods. Additionally, the dataset contains transactional data, including user purchases, offer receptions, views, and completions. Users might make purchases without receiving or viewing offers.
# 
# ### Example
# 
# For instance, consider a scenario where a user receives a discount offer: "Spend $10 and get $2 off" on a Monday. This offer remains valid for 10 days from the day it's received. If the customer makes purchases totaling at least $10 during this validity period, the offer is considered completed.
# 
# However, there are some nuances to be aware of in this dataset. Customers do not actively choose the offers they receive. In other words, a user can receive an offer, never actually open or view it, and still complete the offer. For example, a user might receive the "Spend $10 and get $2 off" offer but never opens it during the 10-day validity period. Despite this, if the customer spends $15 within those ten days, there will be a record indicating offer completion in the dataset. However, it's important to note that the customer wasn't influenced by the offer since they never viewed it.
# 
# ### Cleaning
# 
# Data cleaning becomes particularly crucial and challenging in this context.
# 
# It's essential to consider that certain demographic segments may make purchases even without receiving offers. From a business standpoint, if a customer is likely to make a $10 purchase without any offer, it doesn't make sense to send them a "Spend $10 and get $2 off" offer. Instead, the goal is to understand what specific products or services a particular demographic group typically purchases without the influence of offers.
# 
# ### Final Advice
# 
# Since this is a capstone project, you have the flexibility to analyze the data in any manner you deem appropriate. For instance, you can construct a machine learning model to forecast a person's spending based on their demographic details and the type of offer they receive. Alternatively, you can create a model to predict whether an individual is likely to respond to an offer. However, it's not mandatory to develop a machine learning model at all. You have the option to formulate a set of guidelines or rules that help determine which offer to send to each customer. For instance, you might decide to send "Offer A" to 75 percent of female customers aged 35 who have responded, compared to 40 percent of the same demographic who responded to "Offer B."

# %% [markdown]
# # Data Sets
# 
# The data is distributed across three files:
# 
# **portfolio.json**
# - `id` (string) - Offer ID
# - `offer_type` (string) - Offer type, such as BOGO, discount, or informational
# - `difficulty` (int) - Minimum spending requirement to fulfill an offer
# - `reward` (int) - Reward granted upon successful offer completion
# - `duration` (int) - Offer duration in days
# - `channels` (list of strings) - Communication channels for the offer
# 
# **profile.json**
# - `age` (int) - Customer's age
# - `became_member_on` (int) - Date when the customer registered an app account
# - `gender` (str) - Customer's gender (noting that some entries are marked 'O' for other, rather than 'M' or 'F')
# - `id` (str) - Customer ID
# - `income` (float) - Customer's income
# 
# **transcript.json**
# - `event` (str) - Description of the record (e.g., transaction, offer received, offer viewed)
# - `person` (str) - Customer ID
# - `time` (int) - Time in hours since the start of the test (data commences at time t=0)
# - `value` (dict of strings) - Contains either an offer ID or a transaction amount, depending on the record
# 

# %% [markdown]
# # Business Understanding
# 
# The objective here is to find patterns and show when and where to give specific offer to a specific customer.

# %%
import pandas as pd
import numpy as np
import math
import json

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# %%
# Import the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

# %% [markdown]
# # 1. Data Cleaning
# ## 1.1. Portfolio
# - Rename the 'id' column to 'offer_id'.
# - Split the 'channels' into separate columns.
# - Dissect the 'offer_type' into distinct columns.
# 

# %%
portfolio.head()

# %%
portfolio.describe()

# %%
portfolio.info()

# %%
# Re-name the id column to offer_id.
portfolio.rename(columns={'id':'offer_id'}, inplace=True)

# Split the 'channels' into multiple columns.
channel_dummies = portfolio['channels'].str.join('|').str.get_dummies()
portfolio = pd.concat([portfolio, channel_dummies], axis=1)
portfolio.drop(columns=['channels'], inplace=True)

# Out-put check :
portfolio.head()


# %% [markdown]
# ## 1.2. Profile
# Substitute "id" with "customer_id."
# Rectify the date format.
# Address inconsistencies in the "age" column.
# In both the gender and income columns, there are 2,175 missing values (17,000 - 14,825).

# %%
profile.head()

# %%
profile.info()

# %%
profile.describe()

# %%
# Re-name "id" with "customer_id".
profile.rename(columns={'id':'customer_id'},inplace=True)

# Fix the date.
profile['became_member_on'] = profile['became_member_on'].apply(lambda x: pd.to_datetime(str(x),format='%Y%m%d'))

profile['income'].fillna((profile['income'].mean()), inplace=True)

# Identify customers with irregular ages and exclude them from consideration by introducing a new column called "valid."
ages = profile['age'].unique()
perct_old_ages = profile['age'][profile['age'] > 100].count()/profile['age'].count() * 100
print('''
Unique ages in the df: {},
% customers who has the age > 100: {} %
'''.format(ages, round(perct_old_ages,2)))
profile['valid'] = profile['age'].apply(lambda x: 1 if x <= 100 else 0)

# Out-put check :
profile.head()



# %% [markdown]
# ## 1.3. Transcript
# Update the name of the "person" column to "customer_id."
# Generate dummy variables for the "event" column.
# Expand the values within the "value" column.
# 
# 

# %%
transcript.head()

# %%
transcript.info()

# %%
transcript.describe()

# %%
# Rename the "person" column to "customer_id".
transcript.rename(columns={'person':'customer_id'},inplace=True)

# Unlist the values in "value" column.
transcript['offer_id'] = [list(x.values())[0]  if (list(x.keys())[0] in ['offer_id', 'offer id']) else np.nan for x in transcript['value']]
transcript['amount'] = [list(x.values())[0]  if (list(x.keys())[0] in ['amount']) else np.nan for x in transcript['value']]
transcript.drop(columns=['value'],inplace=True)


# Out-put check :
transcript.head()

# %% [markdown]
# ## 1.4. Merge datasets

# %%
df = pd.merge(transcript, profile, on='customer_id', how="left")
df = pd.merge(df, portfolio, on='offer_id', how="left")
df.head()

# %%
# Streamline the 'offer_id':
offer_ids = df['offer_id'].unique()
cnt = 1
offer_list = {}
for offer in offer_ids:
    offer_list[offer] = 'X'+str(cnt)
    cnt += 1
df['offer_id'] = df['offer_id'].apply(lambda x: offer_list[x] if (x in offer_list.keys()) else x)

# Streamline the customer_id:
customer_ids = profile['customer_id'].unique()
count = 1
customer_list = {}
for cus in customer_ids:
    customer_list[cus] = 'A'+str(count)
    count += 1
df['customer_id'] = df['customer_id'].apply(lambda x: customer_list[x] if (x in customer_list.keys()) else x)

# Add "age_group" for analysis purpose
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 21, 64, 200], 
                        labels=['child', 'teen', 'young adult', 'adult', 'elderly'])

df.head()

# %% [markdown]
# # 2. Analyze
# ## 2.1. Univariate Exploration:
# 
# 1. What constitutes the average income of a Starbucks customer?
# 2. How does the average age of a Starbucks customer look?
# 3. Which promotion among the following is the most prevalent?
# 4. What are the primary values in each column of each dataframe?
# 5. In the context of transcripts, who qualifies as the most devoted customer?

# %% [markdown]
# Let's initiate with the initial inquiry:
# 
# 1. What comprises the mean income of Starbucks customers?

# %%
print('The average income for Starbucks customers: ', round(profile['income'].mean(),2))

# %% [markdown]
# 2. What is the mean age of the typical Starbucks customer?

# %%
print('The average age for Starbucks customers: ', round(profile['age'].mean(),2))

# %% [markdown]
# #### 3. Which of the listed promotions is the most prevalent? ####
# 
# BOGO and Discount appear to be the most frequent, with BOGO slightly edging ahead.

# %%
def addlabels(x,y,rotation='horizontal'):
    '''
    INPUT:
    - x: an array of x labels
    - y: an array of y values
    - rotation: the default is 'horizontal', could be changed to 'vertical' or a number of degree.
    OUTPUT: the label values attached in each bar column.
    '''
    for i in range(len(x)):
        plt.text(i,y[i]//2,y[i],horizontalalignment='center')

# %%


plt.subplot(121)
count_offer_id = df[df['event'] == 'offer completed']['offer_id'].value_counts()
count_offer_id.plot(kind='bar',figsize=(15, 5), rot=45)
plt.xlabel('Offer ID')
plt.ylabel('Count')
plt.title('Distribution of Completed Promotions for each offer')
addlabels(count_offer_id.index, count_offer_id.values);

plt.subplot(122)
count_offer_type = df['offer_type'].value_counts()
count_offer_type.plot(kind='bar',figsize=(15, 5), rot=45)
# plt.xlabel('Offer Type')
plt.ylabel('Count')
plt.title('Offer Type Distribution')
addlabels(count_offer_type.index, count_offer_type.values);



# %% [markdown]
# **4. Which values are the most frequently occurring in each dataframe's columns?**

# %%
plt.subplot(121)
df['age'].hist()
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution');

plt.subplot(122)
count_age_group = df['age_group'].value_counts()
count_age_group.plot(kind='bar',figsize=(15, 5), rot=45)
# plt.xlabel('Age Group')
# plt.ylabel('Count')
plt.title('Age Group Distribution')
addlabels(count_age_group.index, count_age_group.values);



# %%
gender_count = df['gender'].value_counts()
gender_count.plot(kind='bar',figsize=(15, 5), rot=0)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
addlabels(gender_count.index, gender_count.values);

# %% [markdown]
# Most are Adults and Males, interesting...

# %% [markdown]
# **5. Within the context of transcripts, who stands out as the most devoted customer?**

# %%
loyal_customer_count = df[(df['event'] == 'offer completed') | (df['event'] == 'transaction')].groupby(['customer_id', 'event'])['amount'].sum().reset_index()
loyal_customer_count = loyal_customer_count.sort_values('amount',ascending=False).head()

# Visualize
loyal_cus = loyal_customer_count.set_index('customer_id')
loyal_cus.plot(kind='bar',figsize=(15, 5), rot=0)
plt.xlabel('Customer ID')
plt.ylabel('Count')
plt.title('Customer Distribution')
addlabels(loyal_cus.index, loyal_cus['amount']);

for cus in loyal_customer_count['customer_id']:
    print('''
    Profile ID: {},
    Number of Completed Offers: {},
    Amount: {}
    '''.format(cus
               ,df[df['event'] == 'offer completed'].groupby('customer_id')['offer_id'].count().loc[cus]
               ,round(loyal_customer_count[loyal_customer_count['customer_id']==cus]['amount'].values[0], 2))
         )

    

# %% [markdown]
# ## 2.2. Multivariate Exploration
# 
# 1. Among children, teenagers, young adults, adults, and the elderly, which promotion garners the most popularity?
# 2. When examining profiles, who tends to have higher earnings, males or females?
# 3. What types of promotions are favored by each gender?

# %% [markdown]
# **1. Among children, teenagers, young adults, adults, and the elderly, which promotion is the most favored?**

# %%
plt.figure(figsize=(15, 5))
sns.countplot(data=df, x='age_group', hue='offer_type')
plt.title('Offer Distribution by Age Group & Offer Type')
plt.ylabel('Count')
plt.xticks(rotation = 0)
plt.legend(title='Offer Type')
plt.show();

# %% [markdown]
# **2. Who, between males and females, tends to have higher earnings in profile data?**
# 
# Note: Exclude N/A gender entries as they haven't disclosed their gender.

# %%
plt.figure(figsize=(15, 5))
sns.violinplot(x=df[df['gender'] != 'NA']['gender'], y=df['income'])
plt.title('Income vs. Gender')
plt.ylabel('Income')
plt.xlabel('Gender')
plt.xticks(rotation = 0)
plt.show();

# %% [markdown]
# **Note:** The median is represented by the `white dot` in each chart.

# %% [markdown]
# **3. Which types of promotions are favored by each gender?**

# %%
plt.figure(figsize=(15, 5))
sns.countplot(data=df, x=df[df['gender'] != 'NA']['gender'], hue = 'offer_type')
plt.title('Income vs Gender')
plt.ylabel('Income')
plt.xlabel('Gender')
plt.xticks(rotation = 0)
plt.show();

# %% [markdown]
# # 3. Simple Linear Regression Machine Learning Model
# 
# This model aims to predict the **transaction amount** based on consumer age, income, and gender.

# %%
df.head()

# %%
# Extract data :
df_ml = df[['amount','gender','age','income','event']]
df_ml = pd.concat([pd.get_dummies(df_ml['event']), df_ml.drop(columns=['event'])], axis=1)
df_ml = df_ml[['amount','gender','age','income','transaction']]

df_ml = df_ml[(df_ml['transaction'] == 1) & (df_ml['gender'] != 'O')]
df_ml = pd.concat([pd.get_dummies(df_ml['gender']), df_ml.drop(columns=['gender'])], axis=1)
df_ml.drop(columns=['transaction'], axis=1, inplace=True)

df_ml.head()

# %%
# Check NaN values
df_ml.isnull().mean()

# %%
# Define features and target as well as split train/test data
X = df_ml.drop('amount', axis=1)
y = df_ml['amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

scores = dict()

# %%
# No Scaling/Normalization Approach

# Instantiate, Fit & Predict
lr_dumb = LinearRegression(normalize=True) 
lr_dumb.fit(X_train, y_train) 
y_test_preds = lr_dumb.predict(X_test) 

scores['No Scaling/Normalization'] = round(r2_score(y_test, y_test_preds),2)

print(
'''
No Scaling/Normalization: 
r-square score: {} on {} values.'''.format(round(r2_score(y_test, y_test_preds),2), len(y_test))
)

# %%
# Normalization approach

# Fit scaler on the training data
norm = MinMaxScaler().fit(X_train)

# Transform
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

# Instantiate, Fit & Predict
lr_norm = LinearRegression(normalize=True) 
lr_norm.fit(X_train_norm, y_train) 
y_test_preds = lr_norm.predict(X_test_norm) 

scores['Normalization'] = round(r2_score(y_test, y_test_preds),2)

print(
'''
Normalization: 
r-square score: {} on {} values.'''.format(round(r2_score(y_test, y_test_preds),2), len(y_test))
)


# %%
# Scalarization approach

X_train_stand = X_train.copy()
X_test_stand = X_test.copy()

# Apply standardization on numerical features
num_cols = ['age','income']
for i in num_cols:
    # Fit on training data column
    scale = StandardScaler().fit(X_train[[i]])
    
    # Transform
    X_train_stand[i] = scale.transform(X_train_stand[[i]])
    X_test_stand[i] = scale.transform(X_test_stand[[i]])
    
#Instantiate, Fit & Predict
lr_stand = LinearRegression(normalize=True) 
lr_stand.fit(X_train_stand, y_train) 
y_test_preds = lr_stand.predict(X_test_stand)

scores['Scalarization'] = round(r2_score(y_test, y_test_preds),2)

print(
'''
Scalarization: 
r-square score: {} on {} values.'''.format(round(r2_score(y_test, y_test_preds),2), len(y_test))
)



# %%
score_df = pd.DataFrame()
score_df['model type'] = scores.keys()
score_df['r-square value'] = scores.values()
score_df

# %% [markdown]
# ## Conclusion

# %% [markdown]
# The r-squared score remained consistent across all three approaches. Let's revisit the same r-squared score discussed earlier. In a linear regression model, features are not weighted based on their magnitudes as they would be in distance-based methods. Instead, each feature converges toward a minimum in a linear regression model, which operates as a form of gradient descent model. Without scaling, the descent rate and step size for each feature can vary, and this doesn't assign higher weights to larger magnitude features, but it can impact model performance because some features reach the minimum faster than others. Scaling numerical data in a linear regression model is generally advisable to enhance model stability and convergence time. However, as demonstrated in the preceding section, it may not be necessary in terms of feature weighting.
# 
# The second item will delve into the r-squared value, which is expressed on a 0 to 100 percent scale, as explained in the metrics section. A higher percentage indicates a stronger correlation and greater accuracy in model predictions. Consequently, there appears to be a limited connection between the amount spent per transaction by consumers and their age, gender, or annual income in the model described above.

# %% [markdown]
# ## Improvements

# %% [markdown]
# I believe I've reached a point where I've achieved promising results and a reasonable understanding of the data. However, to enhance our outcomes, I intend to refine my data collection process and address any challenges related to NaN values. Additionally, I aim to acquire more information, including the location and timestamp of transactions, as well as details about the store branches and time of day. All of this additional information can play a crucial role in helping us determine when and where to target our offers more effectively.


