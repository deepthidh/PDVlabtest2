
# coding: utf-8

# In[48]:

#oil reserves
import numpy as np
import pandas as pd
from pandas import DataFrame
df=pd.read_csv('oil_reserves.csv')
df


# In[49]:

#region which has maximum oil reserves from 2014 to 2016
df1 = pd.DataFrame(df,columns=['Region','2014','2015','2016'])
df1


# In[50]:

df1.max()


# In[51]:

df1 = pd.DataFrame(df,columns=['2010','2011','2012','2013','2014','2015','2016'])
df1


# In[52]:

df.sum()


# In[53]:

import matplotlib.pyplot as plt

labels = ['2010','2011','2012','2013','2014','2015','2016']
sizes = [1642.42,1681.26,1694.59,1701.56,1706.54,1691.45,1707.27]
explode = (0.2,0.0,0.0,0.0,0.0,0.0,0.0)  

fig1, ax1 = plt.subplots()
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()


# In[ ]:




# In[56]:

#countries with maximum oil reserves in each region
df.loc[df.max(axis=1)][['Region','Country/ Region']]


# In[ ]:




# In[66]:

#oil reserves by each country
df5=df.groupby('Region')[['2010','2011','2013','2014','2015','2016']].sum()
df5


# In[67]:

df5.sum(axis=1)


# In[69]:

import matplotlib.pyplot as plt
import numpy as np

objects = ['Africa','Asia','Europe','Middle east','NAmerica','SAmerica']
x_pos = np.arange(len(objects))
performance = [765.678,291.388,944.091,4787.421,1372.341,1969.592]

plt.bar(x_pos, performance)
plt.xticks(x_pos, objects)
plt.xlabel('Region')
plt.ylabel('Oil reserves')
plt.title('Oil reserves analysis')

plt.show()


# In[72]:

#top 5 countries having maximum reservers in 2015
df.sort_values('2015',ascending=False)[['Country/ Region','2015']].head(5)


# In[74]:

import matplotlib.pyplot as plt

labels = ['Venezuela','SaudiArabia','Canada','Iran','Iraq']
sizes = [300.878,266.578,171.512,158.400,142.503]
explode = (0.0,0.0,0.0,0.0,0.2)

fig1, ax1 = plt.subplots()
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()


# # Milk Production

# In[81]:

import numpy as np
import pandas as pd
from pandas import DataFrame
df7=pd.read_csv('milk_production.csv')
df7


# In[84]:

#state which has maximum milk production in a year 2013-14
df8 = pd.DataFrame(df7,columns=['State/ UT Name','Cow Milk-2013-14','Boffalo Milk-2013-14','Goat Milk-2013-14'])
df8


# In[86]:

df8.max()


# In[93]:

#top 5 milk producing state in each year
df7.sort_values(['Cow Milk-2010-11','Cow Milk-2011-12','Cow Milk-2013-14','Cow Milk-2014-15','Cow Milk-2015-16','Boffalo Milk-2010-11','Boffalo Milk-2011-12','Boffalo Milk-2013-14','Boffalo Milk-2014-15','Boffalo Milk-2015-16','Goat Milk-2010-11','Goat Milk-2011-12','Goat Milk-2013-14','Goat Milk-2014-15','Goat Milk-2015-16'],ascending=False)['State/ UT Name'].head(5)


# In[95]:

#average milk production of all years
df7.mean()


# In[105]:

#line graph to show total production
df9=df7.sum(axis=1)
df9


# In[113]:

df7.mean()


# In[112]:

import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.pyplot import rc
x=df7.sum(axis=1)
x_pts=np.arange(len(x))
plt.plot(x,x_pts,marker='*',label='Milk production')
plt.xlabel("Total milk production")
plt.legend()
plt.ylabel("states")
plt.tight_layout()
plt.show()


# In[114]:

df7.mean()


# In[120]:

#pie chart(subplot)
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

labels = ['2011','2012','2013','2014','2015']
sizes = [1568,1650,1776,1845,2045]
explode = (0.0,0.0,0.0,0.0,0.2)

labels1 = ['2011','2012','2013','2014','2015']
sizes1 = [1889,1806,2012,2075,2123]
explode = (0.0,0.0,0.0,0.0,0.2)

labels2 = ['2011','2012','2013','2014','2015']
sizes2 = [199,217,210,143,149]
explode = (0.0,0.0,0.0,0.0,0.2)


fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.pie(sizes1, explode=explode, labels=labels1, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax3.pie(sizes2, explode=explode, labels=labels2, autopct='%1.1f%%',
        shadow=True, startangle=90)

plt.tight_layout()

plt.show()


# In[ ]:



