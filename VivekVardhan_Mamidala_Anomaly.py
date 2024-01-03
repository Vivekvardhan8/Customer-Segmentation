#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing pandas,Numpy,matplotlib and seaborn libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv('/Users/venkateshmadala/Downloads/Customers-1.csv')
df.head()


# In[5]:


#Drop customer id and gender
df.drop(columns = ['CustomerID','Gender'],axis = 1, inplace = True)


# In[6]:


ax = plt.axes(projection = '3d')
ax.scatter(df['Age'],df['AnnualIncome'],df['SpendingScore']);
ax.set_xlabel('Age')
ax.set_ylabel('AnnualIncome')
ax.set_zlabel('SpendinggScore',rotation = 45)
ax.set_title('Relationship between Age, AnnualIncome and SpendingScore')


# In[7]:


X = df.values

# Using the elbow method to find the optimal value of k
#import KMeans
from sklearn.cluster import KMeans
wcss = [] #WCSS is the sum of squared distance between 
          #each point and the centroid in a cluster
for i in range(1, 11): #range of values you wish to assign k
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) # x is your data
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)


# In[8]:


#plotting elbow method
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('optimum value of k')
plt.ylabel('WCSS')
plt.show()


# In[9]:


# Fitting K-Means with k=6
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
pred = kmeans.fit_predict(X)
#plot
plt.figure(figsize=(10,6))
plt.scatter(X[pred == 0, 0], X[pred == 0, 1], c = 'brown')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], c = 'green')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], c = 'blue')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], c = 'purple')
plt.scatter(X[pred == 4, 0], X[pred == 4, 1], c = 'orange')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 300, c = 'red', marker='o')
plt.title('Customers Clusters ')
plt.xlabel('Spending score')
plt.ylabel('Age')
plt.legend()
plt.show()


# In[10]:


#age and annual income
X = df[['Age','AnnualIncome']].values

# Using the elbow method to find the optimal value of k
#import KMeans
from sklearn.cluster import KMeans
wcss = [] #WCSS is the sum of squared distance between 
          #each point and the centroid in a cluster
for i in range(1, 11): #range of values you wish to assign k
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) # x is your data
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

#plotting elbow method
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('optimum value of k')
plt.ylabel('WCSS')
plt.show()


# In[11]:


# Fitting K-Means with k=2
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
pred = kmeans.fit_predict(X)
#plot
plt.figure(figsize=(10,6))
plt.scatter(X[pred == 0, 0], X[pred == 0, 1], c = 'brown')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], c = 'green')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], c = 'blue')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], c = 'purple')
plt.scatter(X[pred == 4, 0], X[pred == 4, 1], c = 'orange')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 300, c = 'red', marker='o')
plt.title('Age and annual income clusters')
plt.xlabel('Age')
plt.ylabel('Annual income')
plt.legend()
plt.show()


# In[12]:


#age and spending score
X = df[['Age','SpendingScore']].values

# Using the elbow method to find the optimal value of k
#import KMeans
from sklearn.cluster import KMeans
wcss = [] #WCSS is the sum of squared distance between 
          #each point and the centroid in a cluster
for i in range(1, 11): #range of values you wish to assign k
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) # x is your data
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

#plotting elbow method
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('optimum value of k')
plt.ylabel('WCSS')
plt.show()


# In[13]:


# Fitting K-Means with k=4
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
pred = kmeans.fit_predict(X)
#plot
plt.figure(figsize=(10,6))
plt.scatter(X[pred == 0, 0], X[pred == 0, 1], c = 'brown')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], c = 'green')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], c = 'blue')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], c = 'purple')
plt.scatter(X[pred == 4, 0], X[pred == 4, 1], c = 'orange')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 300, c = 'red', marker='o')
plt.title('Age and spending score clusters')
plt.xlabel('Age')
plt.ylabel('Spending score')
plt.legend()
plt.show()


# In[14]:


#Annual Income and Spending Score
X = df[['AnnualIncome','SpendingScore']].values

# Using the elbow method to find the optimal value of k
#import KMeans
from sklearn.cluster import KMeans
wcss = [] #WCSS is the sum of squared distance between 
          #each point and the centroid in a cluster
for i in range(1, 11): #range of values you wish to assign k
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) # x is your data
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

#plotting elbow method
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('optimum value of k')
plt.ylabel('WCSS')
plt.show()


# In[15]:


# Fitting K-Means with k=5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
pred = kmeans.fit_predict(X)
#plot
plt.figure(figsize=(10,6))
plt.scatter(X[pred == 0, 0], X[pred == 0, 1], c = 'brown')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], c = 'green')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], c = 'blue')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], c = 'purple')
plt.scatter(X[pred == 4, 0], X[pred == 4, 1], c = 'orange')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 300, c = 'red', marker='o')
plt.title('Annual income and spending score clusters')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()


# In[16]:


'''For elbow method, where the plot appear to kink is the optimum value of k'''


# In[ ]:




