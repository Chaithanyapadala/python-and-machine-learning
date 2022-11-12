#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_html('https://stats.espncricinfo.com/ci/engine/player/28081.html?class=11;orderby=default;orderbyad=reverse;template=results;type=batting;view=innings')


# In[4]:


for idx,table in enumerate(df):
    print('*'*12)
    print(idx)
    print(table)


# In[5]:


dhoni=df[3]


# In[6]:


dhoni


# In[7]:


dhoni.to_csv(r'C:\Data\Dhoni.csv')


# In[8]:


dhoni['matchtype'] = dhoni['Opposition'].apply(lambda x: x[:4])


# In[9]:



dhoni.drop(columns='Unnamed: 13', inplace=True)


# In[10]:


dhoni.drop(columns='Unnamed: 9', inplace=True)


# In[11]:


dhoni['Start Date']=pd.to_datetime(dhoni['Start Date'])


# In[12]:


dhoni['year'] = dhoni['Start Date'].dt.year.astype(int)


# In[13]:


dhoni['Opposition'] = dhoni['Opposition'].apply(lambda x: x[6:])


# In[14]:


dhoni['Runs'] = dhoni['Runs'].apply(str)
dhoni['not_out'] = np.where(dhoni['Runs'].str.endswith('*'), 1, 0)


# In[15]:


hello = dhoni.loc[((dhoni['Runs'] != 'DNB') & (dhoni['Runs'] != 'TDNB')),'Runs':]


# In[16]:


hello['Runs']=hello.Runs.str.replace('*','')


# In[17]:


hello


# In[18]:


hello['SR']=hello.SR.str.replace('-','0')


# In[203]:


hello['Runs'] = hello['Runs'].astype(int)
hello['BF'] = hello['BF'].astype(int)
hello['SR'] = hello['SR'].astype(float)
hello['4s'] = hello['4s'].astype(int)
hello['6s'] = hello['6s'].astype(int)


# In[ ]:


hello['Mins']=hello.Mins.str.replace('-','0')
hello['Mins'] = hello['Mins'].astype(int)


# In[20]:


hello['SR']=np.round(hello['SR'],2)


# In[21]:


dhoni.shape[0]


# In[22]:


hello['not_out'].sum()


# In[23]:


hello[hello['matchtype']=='ODI '].Runs.max()


# # number matches played against different opposition

# In[297]:


hello['Opposition'].value_counts()


# In[298]:


hello['Opposition'].value_counts().plot(kind='bar', title='Number of matches against different oppositions', figsize=(8, 5));


# # Total runs scored against different opposition

# In[299]:


pd.DataFrame(hello.groupby('Opposition')['Runs'].sum())


# In[300]:


runs_scored_by_opposition = pd.DataFrame(hello.groupby('Opposition')['Runs'].sum())
runs_scored_by_opposition.plot(kind='bar', title='Runs scored against different oppositions', figsize=(8, 5))


# # Total 4s and 6s against different opposition

# In[301]:


pd.DataFrame(hello.groupby('Opposition')['6s','4s'].sum())


# In[302]:


pd.DataFrame(hello.groupby('Opposition')['6s','4s'].sum()).plot(kind='bar', title='Runs scored against different oppositions', figsize=(8, 5))


# In[26]:


hello['Mins']=hello.Mins.str.replace('-','0')
hello['Mins'] = hello['Mins'].astype(int)


# In[27]:





# In[28]:


hello['Mins']


# # Strikerate in differennt matchtypes

# In[69]:


pd.DataFrame(hello.groupby('matchtype')['SR'].mean()).plot(kind='bar');


# # matchestypes played in different years

# In[101]:


plt.figure(figsize=(12,10))
sns.countplot(data=hello,x='year',hue='matchtype');


# In[47]:


t20=pd.DataFrame(hello[hello['matchtype']=='T20I'])
odi=pd.DataFrame(hello[hello['matchtype']=='ODI '])
test=pd.DataFrame(hello[hello['matchtype']=='Test'])



# # Runs scored in all years line chart

# In[104]:


runs_year=hello.groupby(['year']).agg({'Runs':[sum]})
runs_year


# In[106]:


runs_year.plot(kind='line', marker='o', title='Runs scored by year', figsize=(8, 5))
plt.ylabel('runs')


# In[108]:


runs_matchtype=hello.groupby(['matchtype']).agg({'Runs':[sum]})


# In[110]:


runs_matchtype.plot(kind='bar')


# # Runs scored based on dismissal mode

# In[34]:


runs_dismissal=hello.groupby(['Dismissal']).agg({'Runs':[sum]})


# In[113]:


runs_dismissal.plot(kind='bar')


# # Dismissal percentage

# In[303]:


dismissal_mode=hello.Dismissal.value_counts()
dismissal_mode


# In[304]:


plt.figure(figsize=(10,8))
plt.pie(dismissal_mode,labels=['caught','notout','bowled','lbw','run out','stumped'],autopct='%.2f%%')
plt.title('Dismisal modes of Dhoni')
plt.legend(loc='upper left')


# In[258]:


plt.figure(figsize=(30,25))
sns.scatterplot(data = dhoni, x = 'Runs', y= 'year');


# In[39]:


plt.plot(runs_year)


# In[ ]:





# In[ ]:





# # number of times the given runs are scored

# In[280]:


plt.figure(figsize=(11,5))
plt.hist(runs_year, edgecolor = 'b', color= 'orange', bins = 14)
plt.xlabel('runs')
plt.ylabel('Count')


# In[43]:


runs_year


# In[44]:


sns.displot(data = hello, x = hello.Runs, height = 5, aspect = 15/5, kde = True  )


# # correlation between the data values

# In[305]:


plt.figure(figsize=(10,6))
sns.heatmap(hello.corr(), cmap = 'winter', annot=True);


# # Runs  in test odi,test t20 over the years

# In[120]:


runs_test=test.groupby(['year']).agg({'Runs':[sum]})
runs_odi=odi.groupby(['year']).agg({'Runs':[sum]})
runs_t20=t20.groupby(['year']).agg({'Runs':[sum]})


# In[138]:


plt.figure(figsize=(20,10),dpi=150)
plt.plot(runs_test,marker='o',label='TEST')
plt.plot(runs_odi,marker='D',label='ODI')
plt.plot(runs_t20,marker='|',label='T20')
plt.legend()


# In[53]:


test['Runs'].plot(kind='barh', title='runs in each test-match', figsize=(5, 30));


# In[ ]:





# # average of runs over the years

# In[306]:


plt.figure(figsize=(12,5))
pd.DataFrame(test.groupby('year')['Runs'].mean()).plot(kind='line',title='Test');

pd.DataFrame(odi.groupby('year')['Runs'].mean()).plot(kind='line',title='Odi');
pd.DataFrame(t20.groupby('year')['Runs'].mean()).plot(kind='bar',title='T20');


# In[188]:


runs_test=test.groupby(['year']).agg({'Runs':[sum]})
runs_odi=odi.groupby(['year']).agg({'Runs':[sum]})
runs_t20=t20.groupby(['year']).agg({'Runs':[sum]})


# # runs according to sixers 

# In[187]:


pd.pivot_table(data = hello, index='matchtype',columns='6s', 
               values = ['Runs'], 
               aggfunc = [np.sum],
          ).plot(kind='bar', figsize = (15, 7))
plt.legend(loc='best')


# In[ ]:





# In[ ]:





# # Number of centuries against each opposition

# In[307]:


plt.figure(figsize=(15,5))
sns.countplot(data = hello[hello['Runs']>=100],x='Opposition',hue='matchtype')


# # number of fifties against each opposition

# In[308]:


hello[(hello['Runs']>=50) & (hello['Runs']<100)]
plt.figure(figsize=(20,5),dpi=200)
sns.countplot(data = hello[(hello['Runs']>=50) & (hello['Runs']<100)],x='Opposition',hue='matchtype',orient='horizontal')


# # 

# In[227]:


hello[(hello['Runs']>=50) & (hello['Runs']<100)]


# # number of 4's in every year

# In[309]:


hello[hello['4s']>0]
plt.figure(figsize=(20,5))
sns.countplot(data = hello[hello['4s']>0],x='year',hue='matchtype',orient='horizontal')


# # 6 's according to years

# In[310]:


hello[hello['6s']>0]
plt.figure(figsize=(20,5))
sns.countplot(data = hello[hello['6s']>0],x='year',hue='matchtype',orient='horizontal')


# In[311]:


hello[(hello['Runs']>=50) & (hello['Runs']<100)]
plt.figure(figsize=(20,5))
sns.countplot(data = hello[(hello['Runs']>=100)],x='Ground',hue='matchtype')


# # Runs scored at different positions

# In[312]:


pd.pivot_table(data = hello, index='matchtype',columns='Pos', 
               values = ['Runs'], 
               aggfunc = [np.sum],
          ).plot(kind='bar', figsize = (15, 7))
plt.legend(loc='best')


# # highest individual score against different opposition

# In[313]:


plt.figure(figsize=(3,2))
pd.pivot_table(data = hello, index='matchtype',columns='Opposition', 
               values = ['Runs'], 
               aggfunc = [np.max],
          ).plot(kind='bar', figsize = (15, 7))
plt.legend(loc='best')


# In[314]:


pd.pivot_table(data = hello, index='matchtype',columns='Inns', 
               values = ['Runs'], 
               aggfunc = [np.sum],
          ).plot(kind='bar', figsize = (15, 7))
plt.legend(loc='best')


# In[315]:


hello.to_csv(r'C:\Data\Dhoni_stats.csv')


# # Different matchmodes percentage

# In[290]:


matchmode=hello.matchtype.value_counts()
matchmode


# In[316]:


plt.figure(figsize=(10,8))
plt.pie(matchmode,labels=['ODI','Test','T20I'],autopct='%.2f%%')
plt.title('Different matchmode percentage')
plt.legend(loc='upper left')


# In[293]:


hello[hello['matchtype']=='ODI ']


# In[295]:


hello['Opposition']=hello.Opposition.str.replace(' ','')


# In[296]:


hello['Opposition'].value_counts()


# In[ ]:




