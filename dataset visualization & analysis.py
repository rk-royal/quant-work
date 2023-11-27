#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pyreadstat


# In[2]:


df, meta = pyreadstat.read_dta('C:\\Users\\royal\\Downloads\\Poverty and Vulnerability Assessment Tool (PVAT) questionnaire dataset of Nepal 2011_12\\data\\Share pvatData-2011_2012.dta')


# In[3]:


df


# In[4]:


df = df.assign(num_migrated = df["q2_34"])


# In[5]:


df


# In[6]:


df = df[df['num_migrated'] < df['hh_size']]


# In[7]:


df


# In[9]:


df.loc[:, "migration_ratio"] = df["num_migrated"] / df["hh_size"]


# In[10]:


df


# In[12]:


df.loc[:, "help_received"] = np.where( (df["q43d"] == 1) | (df["q43e"] == 1) | (df["q43f"] == 1) | (df["q43g"] == 1) | (df["q43h"] == 1) | (df["q43i"] == 1) | (df["q43j"] == 1),  1,  0)


# In[13]:


df


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[52]:


x = df['hh_size']
y = df['migration_ratio']

data = pd.DataFrame({'Household size': x, 'Migration Ratio': y})

sns.lmplot(x='Household size', y='Migration Ratio', data=data, line_kws={'color': 'red'})
plt.title('Scatter Plot with Regression Line')
plt.show()


# In[51]:


x = df['help_received']
y = df['migration_ratio']

data = pd.DataFrame({'Help Received': x, 'Migration Ratio': y})

sns.lmplot(x='Help Received', y='Migration Ratio', data=data, line_kws={'color': 'red'})
plt.title('Scatter Plot with Regression Line')
plt.show()


# In[17]:


plt.hist(df['migration_ratio'], bins=10, color='skyblue', edgecolor='black')

plt.title('Histogram of Migration Ratio')
plt.xlabel('Migration Ratio')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()


# In[18]:


plt.hist(df['help_received'], bins=10, color='skyblue', edgecolor='black')

plt.title('Histogram of Help Received')
plt.xlabel('Number per household')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()


# In[19]:


df[["hh_size", "migration_ratio", "help_received"]].describe()


# In[20]:


df["q5eT"]


# In[21]:


plt.hist(df['q5eT'])

plt.show()


# In[22]:


df['q5eT'].describe


# In[26]:


x = df['hh_age']
y = df['migration_ratio']

data = pd.DataFrame({'Age': x, 'Migration Ratio': y})

sns.lmplot(x='Age', y='Migration Ratio', data=data, line_kws={'color': 'red'})
plt.title('Scatter Plot with Regression Line')
plt.show()


# In[46]:


x = df['hh_sex']
y = df['migration_ratio']

data = pd.DataFrame({'Sex': x, 'Migration Ratio': y})

sns.lmplot(x='Sex', y='Migration Ratio', data=data, line_kws={'color': 'red'})
plt.title('Scatter Plot with Regression Line')
plt.show()


# In[47]:


x = df['hh_martial']
y = df['migration_ratio']

data = pd.DataFrame({'Marital Status': x, 'Migration Ratio': y})

sns.lmplot(x='Marital Status', y='Migration Ratio', data=data, line_kws={'color': 'red'})
plt.title('Scatter Plot with Regression Line')
plt.show()


# In[48]:


x = df['region']
y = df['migration_ratio']

data = pd.DataFrame({'Region': x, 'Migration Ratio': y})

sns.lmplot(x='Region', y='Migration Ratio', data=data, line_kws={'color': 'red'})
plt.title('Scatter Plot with Regression Line')
plt.show()


# In[49]:


x = df['q5eT']
y = df['migration_ratio']

data = pd.DataFrame({'Time to reach a paved road': x, 'Migration Ratio': y})

sns.lmplot(x='Time to reach a paved road', y='Migration Ratio', data=data, line_kws={'color': 'red'})
plt.title('Scatter Plot with Regression Line')
plt.show()


# In[54]:


df[["hh_size", "hh_sex", "hh_age", "region", "hh_martial", "migration_ratio", "help_received"]].describe()


# In[67]:


df.loc[9, 'q2_2d3']


# In[70]:


is_alphabetical = df['q2_2d3'].apply(lambda x: isinstance(x, str))

df['is_alphabetical'] = is_alphabetical
print(df)


# In[89]:


# Define a function to convert numeric values to NaN
def convert_to_nan(value):
    try:
        float_value = float(value)
        return np.nan
    except ValueError:
        return value

# Specify the columns you want to apply the conversion to
selected_columns = ['q2_5a3', 'q2_5a4', 'q2_5b1','q2_5b2','q2_5b3','q2_5b4','q2_5c1','q2_5c2','q2_5c3','q2_5c4','q2_5d1','q2_5d2','q2_5d3','q2_5d4','q2_5e1','q2_5e2','q2_5e3']

# Apply the function to selected columns
df[selected_columns] = df[selected_columns].applymap(convert_to_nan)

# Display the updated DataFrame
print(df)


# In[90]:


df["q2_5a3"]


# In[91]:


# Define the columns you want to consider
selected_columns = ['q2_5a3', 'q2_5a4', 'q2_5b1','q2_5b2','q2_5b3','q2_5b4','q2_5c1','q2_5c2','q2_5c3','q2_5c4','q2_5d1','q2_5d2','q2_5d3','q2_5d4','q2_5e1','q2_5e2','q2_5e3']

# Concatenate selected columns into a single Series
selected_text = df[selected_columns].apply(lambda col: ' '.join(col.astype(str)), axis=1)

# Split the text into words
selected_words = ' '.join(selected_text).split()

# Create a Series with word frequencies
word_frequencies = pd.Series(selected_words).value_counts()

# Display the frequencies
print("Word Frequencies in Selected Columns:")
print(word_frequencies)


# In[1]:


df


# In[ ]:




