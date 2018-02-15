#graphing animation
#import matplotlib.animation as animation
import pandas as pd
from sklearn.cluster import KMeans
import sklearn.metrics as sm
#get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
#from sqlalchemy import create_engine


# In[2]:
#engine = create_engine("mysql+mysqldb://root:"+'lbr123'+"@localhost/student")


# In[3]:


#df = pd.read_sql('SELECT * FROM main', con=engine, index_col=['index'])

df = pd.read_csv("student-mat_new.csv",sep=";")

# In[60]:


#clusters = input("Enter Number of clusters?")
clusters = 3
max_iter = 500
instance = 0


# In[4]:


#df = pd.read_csv("student-mat_new.csv",sep=";")
df.head()


# In[5]:


#df.to_sql(con=engine, name='main', if_exists='replace')


# In[6]:


df.columns


# In[7]:


df.describe()


# #remove the extra features from datasets
# del df['G1']
# del df['G2']
# del df['G3']

# ## Encoding the categorical variable 

# In[8]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
df['school']=labelencoder_X.fit_transform(df['school'])
df['sex']=labelencoder_X.fit_transform(df['sex'])
df['address']=labelencoder_X.fit_transform(df['address'])
df['Pstatus']=labelencoder_X.fit_transform(df['Pstatus'])
df['Mjob']=labelencoder_X.fit_transform(df['Mjob'])
df['Fjob']=labelencoder_X.fit_transform(df['Fjob'])
df['schoolsup']=labelencoder_X.fit_transform(df['schoolsup'])
df['famsup']=labelencoder_X.fit_transform(df['famsup'])
df['paid']=labelencoder_X.fit_transform(df['paid'])
df['activities']=labelencoder_X.fit_transform(df['activities'])
df['nursery']=labelencoder_X.fit_transform(df['nursery'])
df['higher']=labelencoder_X.fit_transform(df['higher'])
df['internet']=labelencoder_X.fit_transform(df['internet'])
df['romantic']=labelencoder_X.fit_transform(df['romantic'])
df['higher']=labelencoder_X.fit_transform(df['higher'])
df['famsize']=labelencoder_X.fit_transform(df['famsize'])
df['reason']=labelencoder_X.fit_transform(df['reason'])
df['guardian']=labelencoder_X.fit_transform(df['guardian'])


# In[9]:


df.head()


# ## Generating rendom values

# In a given dataset data for Skills and CGPA is missing. So, I have generated values for it 

# In[10]:


CGPA = np.random.uniform(low=1, high=4, size=(395,))
skills= list(np.random.choice([0, 1], size=(395,)))
#adding CGPA and skills in df 
df['CGPA'] =CGPA
df['skills'] = skills


# # Removing extra dataset

# In[11]:


temp = df[['sex','age','failures','paid','activities','famsup','health','absences','CGPA','skills']]
temp.head()


# In[12]:


success = []
for i in temp['CGPA']:
    if(i>=2):
        success.append(1)
    else:
        success.append(0)    


# In[13]:


temp['success'] = success


# In[14]:


temp.head()


# In[15]:


y = temp['success']
del temp['success']
x = temp


# In[16]:


kmean = KMeans(n_clusters=int(clusters),random_state=instance,max_iter=max_iter)


# In[17]:


kmean.fit(x.as_matrix())


# In[18]:


colorMap=np.array(['Red','Blue','Green','Orange',"Gray"])


# In[19]:


plt.scatter(x=temp['CGPA'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("CGPA vs Success")
plt.xlabel("Success")
plt.ylabel("CGPA")
plt.show()



# In[20]:


show = pd.DataFrame()
show['CGPA'] = temp['CGPA']
show['Success'] = y
show['Clusters'] = kmean.labels_
#------------------- updated code ------------------------------------
ck=show['Clusters'].unique()
suc=show['Success'].unique()
#print(len(show['Clusters'] == 0))
for i in ck:
	for j in suc:
		print("Cluster: "+str(i)+" Success: "+str(j))
		print(str(len(show[(show['Clusters'] == i) & (show['Success']==j)])/len(show['Clusters'] == i)*100)+"%")
print(show)
#------------------- updated code ------------------------------------

# In[21]:
"""

plt.scatter(x=temp['famsup'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("FamilySupport vs Success")
plt.xlabel("FamilySupport")
plt.ylabel("Success")
plt.show()


# In[22]:


show = pd.DataFrame()
show['famsup'] = temp['famsup']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show)

# In[23]:


plt.scatter(x=temp['absences'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("absences vs Success")
plt.xlabel("absences")
plt.ylabel("Success")
plt.show()


# In[24]:


show = pd.DataFrame()
show['absences'] = temp['absences']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show)

# In[25]:


plt.scatter(x=temp['skills'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("skills vs Success")
plt.xlabel("skills")
plt.ylabel("Success")
plt.show()


# In[26]:


show = pd.DataFrame()
show['skills'] = temp['skills']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show)

# In[27]:


plt.scatter(x=temp['paid'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("Paid vs Success")
plt.xlabel("paid")
plt.ylabel("Success")
plt.show()


# In[28]:


show = pd.DataFrame()
show['paid'] = temp['paid']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show)

# In[29]:


plt.scatter(x=temp['age'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("Age vs Success")
plt.xlabel("age")
plt.ylabel("Success")
plt.show()


# In[30]:


show = pd.DataFrame()
show['age'] = temp['age']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show)

# In[31]:


plt.scatter(x=temp['sex'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("Gender vs Success")
plt.xlabel("Gender")
plt.ylabel("Success")
plt.show()


# In[32]:


show = pd.DataFrame()
show['gender'] = temp['sex']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show)

# In[33]:


plt.scatter(x=temp['health'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("health vs Success")
plt.xlabel("health")
plt.ylabel("Success")
plt.show()


# In[34]:


show = pd.DataFrame()
show['health'] = temp['health']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show)

# In[35]:


plt.scatter(x=temp['activities'],y=y,c = colorMap[kmean.labels_],s=40)
plt.title("activities vs Success")
plt.xlabel("activities")
plt.ylabel("Success")
plt.show()


# In[36]:


show = pd.DataFrame()
show['activities'] = temp['activities']
show['Success'] = y
show['Clusters'] = kmean.labels_


print(show)


# In[37]:


plt.scatter(x=temp['CGPA'],y=temp['age'],c = colorMap[kmean.labels_],s=40)
plt.title("CGPA vs age")
plt.xlabel("CGPA")
plt.ylabel("age")
plt.show()


# ## Bar Chart

# In[50]:


show = pd.DataFrame()
show['age'] = temp['age']
show['Success'] = y
show['Clusters'] = kmean.labels_
print(show.groupby('Clusters').size())


# In[59]:


show = pd.DataFrame()
show['Clusters'] = kmean.labels_
#show['Clusters'].count()

ax = show.groupby('Clusters').size().plot(kind='bar', title ="Values in each clusters", figsize=(10, 5), legend=True, fontsize=12)
ax.set_xlabel("Clusters", fontsize=12)
ax.set_ylabel("Values", fontsize=12)
plt.show()


# ## Results

# In[ ]:


print("Clusters# = "+clusters)
print("Maximum Iteration=  "+str(max_iter))
print("Num of instence="+str(instance))

accuracy = float(metrics.accuracy_score(y, kmean.labels_))
print ("Accuracy = "+str(accuracy*100)+"%")
"""
