#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (7)\Data.csv")


# In[7]:


df


# In[124]:


x = df[['Country','Age','Salary']].values


# In[125]:


x


# In[11]:


y=df[['Purchased']].values
y


# In[13]:


df['Purchased'].replace({'No':'1','Yes':'2'},inplace = True)
df


# In[128]:


from sklearn.impute import SimpleImputer


# In[129]:


imp = SimpleImputer(missing_values=np.nan,strategy='mean')


# In[130]:


imp = imp.fit(x[:,1:3])


# In[131]:


x[:,1:3] = imp.transform(x[:,1:3])


# In[132]:


x


# In[133]:


y  = df[['Purchased']].values


# In[134]:


y


# In[135]:


mean_age = df.Age.mean()
df.Age = df.Age.fillna(mean_age)
df.head(8)


# In[139]:


mean_salary = df.Salary.mean()
df.Salary = df.Salary.fillna(mean_salary)
df


# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


labelencoder = LabelEncoder()


# In[28]:


x[:,0] = labelencoder.fit_transform(x[:,0])


# In[29]:


x


# In[30]:


from sklearn.preprocessing import OneHotEncoder


# In[31]:


onehotencoder = OneHotEncoder()


# In[32]:


x = onehotencoder.fit_transform(df.Country.values.reshape(-1,1)).toarray()


# In[33]:


x


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[36]:


x_train


# In[37]:


y_train


# In[38]:


x_test


# In[39]:


from sklearn.preprocessing import StandardScaler


# In[40]:


sc = StandardScaler()


# In[41]:


x_train = sc.fit_transform(x_train)


# In[42]:


x_test = sc.fit_transform(x_test)


# In[43]:


x_train


# In[44]:


x_test


# In[45]:


df = pd.read_csv(r"C:\Users\hplap\OneDrive\Desktop\Insurance.csv")


# In[46]:


df


# In[47]:


x = df[['Age','Hieght','Weight']].values
x


# In[48]:


y = df[['Premium']].values
y


# In[49]:


from sklearn.impute import SimpleImputer


# In[50]:


imp = SimpleImputer(missing_values=np.nan,strategy='mean')


# In[51]:


imp = imp.fit(x[:,1:3])


# In[52]:


x[:,1:3]= imp.transform(x[:,1:3])
x


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[55]:


(len(x_train))


# In[56]:


len(x_test)


# In[57]:


y_train


# In[58]:


y_test


# In[59]:


sns.lmplot (x ='Age',y = 'Premium',data = df)
plt.show()


# In[60]:


sns.barplot (x ='Age',y = 'Premium',data = df)
plt.show()


# In[61]:


df


# In[62]:


sns.lmplot(x='Age',y="Premium",data = df)
plt.show()


# In[63]:


import pandas as pd
df = pd.read_csv(r"C:\Users\hplap\Downloads\Book1.csv")
df


# In[64]:


x= df[['Age']].values
x


# In[65]:


df['Bought Insurance'].replace({'no':'1','yes':'2'},inplace = True)


# In[66]:


df


# In[67]:


plt.scatter(x="Age",y="Bought Insurance",data = df)
plt.show()


# In[68]:


sns.barplot(x="Age",y = "Bought Insurance",data = df)
plt.show()


# In[69]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (8)\Iris.csv")
df


# In[70]:


df['Species'].replace({'Iris-setosa':'1','Iris-virginica':'2','Iris-vercicolor':'3'},inplace = True)


# In[153]:


df


# In[71]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
from sklearn.model_selection import train_test_split


# In[72]:


x_train,_x_test,y_train,y_test=train_test_split(df[['SepalLengthCm','SepalWidthCm',
                                                    'PetalLengthCm','PetalWidthCm']],df['Species'],test_size = 0.2)


# In[73]:


len(x_train)


# In[74]:


len(y_test)


# In[75]:


from sklearn.linear_model import LogisticRegression


# In[76]:


lr = LogisticRegression()
lr


# In[77]:


lr.fit(x_train,y_train)


# In[78]:


lr.predict(x_train)


# In[79]:


x_train


# In[80]:


lr.score(x_train,y_train)


# In[81]:


import seaborn as sns


# In[82]:


sns.pairplot(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']],hue = 'Species')
plt.show()


# In[91]:


import pandas as pd


# In[92]:


data = pd.read_csv(r"C:\Users\hplap\Downloads\tennis\PlayTennis.csv")
data


# In[93]:


from sklearn.preprocessing import LabelEncoder
Outlook = LabelEncoder()
Temperature  = LabelEncoder()
Humidity = LabelEncoder()
Wind = LabelEncoder()
Play = LabelEncoder()


# In[94]:


data["Outlook"] = Outlook.fit_transform(data["Outlook"])
data["Temperature"] = Outlook.fit_transform(data["Temperature"])
data["Humidity"] =Outlook.fit_transform(data["Humidity"])
data["Wind"] = Outlook.fit_transform(data["Wind"])
data["Play"] = Outlook.fit_transform(data["Play"])


# In[95]:


data


# In[96]:


feature_cls = ['Outlook','Temperature','Humidity','Wind']
x = data[feature_cls]
y = data.Play


# In[97]:


x


# In[98]:


y


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[101]:


x_train


# In[102]:


from sklearn.tree import DecisionTreeClassifier


# In[103]:


classifier = DecisionTreeClassifier(criterion='entropy')


# In[104]:


classifier


# In[105]:


classifier.fit(x_train,y_train)


# In[106]:


classifier.predict(x_test)


# In[107]:


classifier.score(x_test,y_test)


# In[108]:


from sklearn import tree
tree.plot_tree(classifier)


# In[147]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import linear_model


# In[148]:


df = pd.read_csv(r"C:\Users\hplap\OneDrive\Desktop\Insurance_1.csv")
df


# In[149]:


mean_height = df.Height.mean()
df.Height = df.Height.fillna(mean_height)
df


# In[150]:


reg = linear_model.LinearRegression()


# In[152]:


reg.fit(df[['Age','Height','Weight']],df['Premium'])


# In[153]:


reg.coef_


# In[154]:


reg.intercept_


# In[162]:


reg.predict([[28,170.62,85]])


# In[163]:


2150.26052416*28+-248.45851574*170.62+312.65291961*85+-16827.013154824977


# In[164]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\Position_Salaries.csv")
df


# In[166]:


x = df.iloc[:,1:2].values
x


# In[167]:


y = df.iloc[:,2].values
y


# In[175]:


from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns


# In[178]:


plt.scatter(x='Position',y = 'Salary',data = df)
plt.show()


# In[181]:


sns.lmplot(x= 'Level',y = 'Salary',data = df)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[170]:


reg = linear_model.LinearRegression()


# In[171]:


reg.fit(x,y)


# In[183]:


reg.predict([[6.8]])


# In[184]:


from sklearn.preprocessing import PolynomialFeatures


# In[185]:


reg = PolynomialFeatures(degree = 2)


# In[ ]:


v = reg.fit_transform(x)


# In[192]:


reg2 = linear_model.LinearRegression()


# In[195]:


reg2.fit(v,y)


# In[199]:


reg2.predict(reg.fit_transform([[6.5]]))


# In[211]:


from sklearn.preprocessing import PolynomialFeatures


# In[220]:


re = PolynomialFeatures(degree = 2)
re


# In[221]:


b = re.fit_transform(x)


# In[216]:


c = linear_model.LinearRegression()


# In[222]:


c.fit(b,y)


# In[223]:


c.predict(re.fit_transform([[9.2]]))


# In[230]:


from sklearn.preprocessing import PolynomialFeatures


# In[233]:


a = PolynomialFeatures(degree = 2)
a


# In[236]:


b = a.fit_transform(x)


# In[237]:


c = linear_model.LinearRegression()


# In[238]:


c.fit(b,y)


# In[240]:


c.predict(a.fit_transform([[8.2]]))


# In[241]:


from sklearn.preprocessing import PolynomialFeatures


# In[243]:


a = PolynomialFeatures(degree = 2)
a


# In[244]:


b = a.fit_transform(x)


# In[245]:


c = linear_model.LinearRegression()


# In[246]:


c.fit(b,y)


# In[247]:


c.predict(a.fit_transform([[3.2]]))


# In[259]:


from sklearn.preprocessing import PolynomialFeatures


# In[260]:


ab = PolynomialFeatures(degree = 2)
ab


# In[261]:


de = ab.fit_transform(x)


# In[262]:


ff = linear_model.LinearRegression()


# In[263]:


ff.fit(de,y)


# In[265]:


ff.predict(ab.fit_transform([[8.4]]))


# In[28]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import seaborn as sns 


# In[29]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\Book1.csv")
df


# In[30]:


df['Bought Insurance'].replace({'no':'0','yes':'1'},inplace = True)


# In[31]:


df


# In[32]:


plt.scatter(x='Age',y = 'Bought Insurance',data = df)
plt.show()


# In[33]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train,x_test,y_train,y_test = train_test_split(df[['Age']],df['Bought Insurance'],test_size = 0.2)


# In[43]:


x_train


# In[44]:


y_train


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


reg = LogisticRegression()


# In[47]:


reg.fit(x_train,y_train)


# In[48]:


reg.predict(x_test)


# In[49]:


reg.predict([[60]])


# In[50]:


reg.predict([[50]])


# In[52]:


reg.predict([[29]])


# In[53]:


reg.score(x_train,y_train)


# In[55]:


data = pd.read_csv(r"C:\Users\hplap\Downloads\archive (8)\Iris.csv")
data


# In[57]:


data['Species'].unique()


# In[58]:


data['Species'].replace({'Iris-setosa':'1','Iris-versicolor':'2','Iris-virginica':'3'},inplace = True)


# In[59]:


data


# In[63]:


from sklearn.model_selection import train_test_split


# In[67]:


x_train,x_test,y_train,y_test = train_test_split(data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],data['Species'],test_size = 0.2)


# In[68]:


len(x_train)


# In[69]:


len(x_test)


# In[71]:


from sklearn.linear_model import LogisticRegression


# In[73]:


la = LogisticRegression()
la


# In[74]:


la.fit(x_train,y_train)


# In[ ]:





# In[77]:


la.predict(x_test)


# In[80]:


x_test


# In[83]:


la.score(x_train,y_train)


# In[104]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as plt


# In[105]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (6)\Salary_Data.csv")


# In[106]:


df


# In[107]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[108]:


x


# In[109]:


y


# In[110]:


from sklearn.model_selection import train_test_split


# In[123]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[124]:


x_train


# In[125]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[126]:


reg = LinearRegression()


# In[127]:


reg.fit(x_train,y_train)


# In[128]:


y_pred = reg.predict(x_test)
x_pred = reg.predict(x_train)


# In[132]:


print("Training set")
plt.scatter(x_train,y_train,color = 'green')
plt.plot(x_train,x_pred,color = 'red')
plt.show()


# In[137]:


print("test set")
plt.scatter(x_test,y_test,color = 'blue')
plt.plot(x_train,x_pred, color = 'red')
plt.show()


# In[138]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (7)\User_Data.csv")


# In[139]:


df


# In[159]:


x = df.iloc[:,[2,3]].values


# In[160]:


y = df.iloc[:,4].values


# In[161]:


x


# In[162]:


y


# In[163]:


from sklearn.model_selection import train_test_split


# In[164]:


x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2)


# In[166]:


from sklearn.preprocessing import StandardScaler


# In[167]:


st = StandardScaler()


# In[172]:


x_train = st.fit_transform(x_train)


# In[173]:


x_test = st.fit_transform(x_test)


# In[170]:


from sklearn.linear_model import LogisticRegression


# In[171]:


cf = LogisticRegression()


# In[174]:


cf.fit(x_train,y_train)


# In[176]:


y_pred = cf.predict(x_test)


# In[177]:


y_pred


# In[178]:


from sklearn.metrics import confusion_matrix


# In[179]:


cf = confusion_matrix(y_pred,y_test)


# In[180]:


cf


# # Random Forest Algo

# In[185]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (7)\User_Data.csv")
df.head()


# In[186]:


x = df[['Age','EstimatedSalary']]
y = df['Purchased']


# In[187]:


from sklearn.model_selection import train_test_split


# In[188]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[189]:


from sklearn.ensemble import RandomForestClassifier


# In[190]:


clf = RandomForestClassifier()


# In[191]:


clf.fit(x_train,y_train)


# In[195]:


y_pred = clf.predict(x_test)


# In[193]:


from sklearn.metrics import accuracy_score


# In[196]:


accuracy_score(y_test,y_pred)


# In[ ]:





# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (7)\User_Data.csv")
df.head()


# In[13]:


x = df.iloc[:,[2,3]].values
y = df.iloc[:,4].values


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train = st.fit_transform(x_train)
y_test = st.fit_transform(x_test)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)


# In[20]:


classifier.fit(x_train, y_train)


# In[22]:


pred = classifier.predict(x_test)


# In[23]:


pred


# In[24]:


from sklearn.svm import SVC


# In[26]:


ar = SVC(kernel = 'linear')


# In[27]:


ar


# In[28]:


ar.fit(x_train,y_train)


# In[29]:


pred2 = ar.predict(x_test)


# In[30]:


pred2


# In[31]:


from sklearn.metrics import accuracy_score


# In[33]:


accuracy_score(pred,pred2)


# In[100]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split


# In[101]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (10)\weatherAUS.csv")
df.head()


# In[102]:


print('Size of wheather data frame is :',df.shape)


# In[103]:


print(df[0:5])


# In[104]:


# Data preprocessing 
df.count().sort_values()


# In[105]:


df.isnull().sum()


# In[106]:


df.drop(columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','Date'],axis = 1)


# In[109]:


df = df.dropna(how = 'any')
df


# In[110]:


df.shape


# In[90]:


from scipy import stats


# In[91]:


z = np.abs(stats.zscore(df._get_numeric_data()))


# In[92]:


print(z)


# In[49]:


df = df[(z<3).all(axis = 1)]


# In[50]:


print(df.shape)


# In[51]:


df['RainToday'].replace({'No ':'0','Yes ':'1'},inplace = True)
df['RainTomorrow'].replace({'No':'0','Yes':'1'},inplace = True)


# In[93]:


catagorical_columns  =['WindGustDir','WindDir3pm','WindDir9am']


# In[94]:


for col in catagorical_columns:
    print(np.unique(df[col]))


# In[54]:


df = pd.get_dummies(df,columns = column)
print(df.iloc[4:9])


# In[67]:


from sklearn import preprocessing


# In[69]:


scaler =preprocessing.MinMaxScaler()


# In[75]:


ar = scaler.fit


# In[76]:


df = pd.DataFrame(ar.transform(df),index = df.index, columns = df.column)


# In[ ]:




