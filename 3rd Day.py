#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np


# In[27]:



from numpy import random

x = random.normal(size=(2, 3))

print(x)


# In[28]:



y = random.normal(loc = 2, scale = 5, size = (6,7))
y

import numpy as np
m1 = np.mean(y)
m2 = np.median(y)
m3 = np.median(y)

print(m1, m2, m3)


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.normal(size=(42)))
plt.show()


# In[31]:


z = pd.DataFrame(y).to_csv("Normal Gauss")


# In[32]:


pd.read_csv("Normal Gauss")


# In[33]:


#z score of the distribution

from scipy.stats import norm
p=.95
value=norm.ppf(p)
print(value)


# In[34]:



from statsmodels.stats.weightstats import ztest as ztest
data = (1,2,3,4,5,6,7,8,9,1,2,12,12,33,44,44,55,666,77,66,33,33,34,66,77,77,88,99,33,545,65,545,65,65,76,8788,78,8767,76,67,65,56)
len(data)

x,y=ztest(data, value=42)
print("Citical value", x, "P-value", y)

if y < 0.05:    # alpha value is 0.05 or 5%
   print(" we are rejecting null hypothesis")
else:
  print("we are accepting null hypothesis")


# In[35]:



from scipy.stats import ttest_1samp
import numpy as np

#10 ages and you are checking whether avg age is 30 or not.
#H0: The average age is 30
#H1: The average age is not 30.
ages = np.array([32,34,29,29,22,39,38,37,38,36,30,26,22,22])
print(ages)
#mean of the age 
ages_mean = np.mean(ages)
print(ages_mean)
#One Sample t-test
tset, pval = ttest_1samp(ages, 30)
print('p-values',pval)
if pval < 0.05:    # alpha value is 0.05 or 5%
   print(" we are rejecting null hypothesis")
else:
  print("we are accepting null hypothesis")


# In[36]:


age_x=[25,26,27,29,33,32,35,28,35,39]
devop=[3525,3326,3627,3129,3633,3032,3115,3628,3435,3009]
javadop=[4525,5326,5627,5129,5633,5032,6115,6628,6435,6009]
age_x.sort()
devop.sort()
javadop.sort()


# In[37]:



from matplotlib import pyplot as plt
plt.style.available

plt.plot(age_x,devop,label="All devep")
plt.plot(age_x,javadop,label="All Java ")
plt.xlabel("Ages")
plt.ylabel("Median Salary (USD)")
plt.legend()


# In[38]:


age1 = np.random.normal([60, 50, 22])
salary2 = np.random.normal([4000, 5000, 6000])
salary3 = np.random.normal([7000, 8000, 9000])

age1.sort()
salary2.sort()
salary3.sort()


# In[39]:


plt.plot(age1,salary2,label="All salary1")
plt.plot(age1,salary3,label="All salary2 ")
plt.xlabel("Ages")
plt.ylabel("Median Salary (USD)")
plt.legend()


# In[40]:





# In[ ]:




