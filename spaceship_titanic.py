import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

#import dataset from github raw data
df_st=pd.read_csv("train.csv")

#display the first 5 rows of the dataset
df_st['Destination'].count()
df_st.head()
df_st['Destination'].info()
df_st.info()

#Remove the columns model will not use

df_st.drop(['Name'],axis=1,inplace=True)


sns.pairplot(df_st)
sns.pairplot(df_st[['Transported','CryoSleep']],dropna=True)


#'PassengerId' Destination Age VIP RoomService FoodCourt Spa Name
#'HomePlanet' , 'CryoSleep' , 'Cabin' , 'ShoppingMall' VRDeck Transported

#df_st.groupby('Transported').mean(numeric_only=True) #group by the target variable and calculate the mean of the other variables

#data observing
df_st['CryoSleep'].value_counts()
df_st['RoomService'].value_counts()


#handdle missing data
df_st.isnull().sum().sort_values(ascending=False)
len(df_st)

df_st['CryoSleep'].fillna(df_st['CryoSleep'].value_counts().idxmax(),inplace=True)
#if 'CryoSleep' is True, then 'RoomService', 'FoodCourt', 'Spa' ,'VRDeck','ShoppingMall' should be filling 0
#fill the missing value of 'RoomService' with the value 0, when 'CryoSleep' is True
df_st.loc[df_st['CryoSleep'] == True, 'RoomService'] = df_st.loc[df_st['CryoSleep'] == True, 'RoomService'].fillna(0)
#fill the missing value of 'RoomService' with the median, when 'CryoSleep' is False
df_st['RoomService'].fillna(df_st['RoomService'].median(),inplace=True)

df_st.loc[df_st['CryoSleep'] == True, 'FoodCourt'] = df_st.loc[df_st['CryoSleep'] == True, 'FoodCourt'].fillna(0)
df_st['FoodCourt'].fillna(df_st['FoodCourt'].median(),inplace=True)
df_st.loc[df_st['CryoSleep'] == True, 'Spa'] = df_st.loc[df_st['CryoSleep'] == True, 'Spa'].fillna(0)
df_st['Spa'].fillna(df_st['Spa'].median(),inplace=True)
df_st.loc[df_st['CryoSleep'] == True, 'VRDeck'] = df_st.loc[df_st['CryoSleep'] == True, 'VRDeck'].fillna(0)
df_st['VRDeck'].fillna(df_st['VRDeck'].median(),inplace=True)
df_st.loc[df_st['CryoSleep'] == True, 'ShoppingMall'] = df_st.loc[df_st['CryoSleep'] == True, 'ShoppingMall'].fillna(0)
df_st['ShoppingMall'].fillna(df_st['ShoppingMall'].median(),inplace=True)

#deal with HomePlanet ,filling the missing value with the most frequent value
#using get_dummies' to 'HP_mars' ,'HP_euro' and 'HP_earth'
df_st['HomePlanet'].fillna(df_st['HomePlanet'].value_counts().idxmax(),inplace=True)
df_st = pd.get_dummies(data=df_st,dtype=int,columns=['HomePlanet'])


#deal with 'Cabin' ,split to 'Cabin_lvl','Cabin_num','Cabin_ps'
df_st[['Cabin_lvl', 'Cabin_num', 'Cabin_ps']] = df_st['Cabin'].str.split('/', expand=True)
df_st.drop("Cabin",axis=1,inplace=True)
df_st = pd.get_dummies(data=df_st,dtype=int,columns=['Cabin_ps'])
df_st.drop(['Cabin_ps_S'],axis=1,inplace=True)
df_st['Cabin_ps_P'].fillna(df_st['Cabin_ps_P'].value_counts().idxmax(),inplace=True)
df_st['Cabin_num'].fillna(df_st['Cabin_num'].value_counts().idxmax(),inplace=True)
df_st['Cabin_lvl'] = df_st['Cabin_lvl'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8})
df_st['Cabin_lvl'].fillna(df_st['Cabin_lvl'].value_counts().idxmax(),inplace=True)

#deal with 'Age' ,filling the missing value with the median
df_st['Age'].fillna(df_st['Age'].median(),inplace=True)

#deal with 'VIP' ,filling the missing value with the most frequent value
df_st['VIP'].fillna(df_st['VIP'].value_counts().idxmax(),inplace=True)
df_st = pd.get_dummies(data=df_st,dtype=int,columns=['VIP'])

#deal with 'Destination' ,filling the missing value with the most frequent value
df_st['Destination'] = df_st['Destination'].map({'TRAPPIST-1e':1, '55 Cancri e':2, 'PSO J318.5-22':3})
df_st['Destination'].fillna(df_st['Destination'].value_counts().idxmax(),inplace=True)
#deal with 'CryoSleep'
df_st['CryoSleep'] = df_st['CryoSleep'].map({True:1,False:0})


df_st.corr()['Transported'].sort_values(ascending=False)

X = df_st.drop(['PassengerId','Transported'],axis=1)
Y = df_st['Transported']



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=67)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=5000) #create a model
lr.fit(X_train,Y_train)

predictions = lr.predict(X_test)
#evaluation
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
accuracy_score(Y_test,predictions)
recall_score(Y_test,predictions)
precision_score(Y_test,predictions)
confusion_matrix(Y_test,predictions)

#pd.DataFrame(confusion_matrix(Y_test,predictions),columns=['Predicted not Survived','Predicted Survived'],index=['True not Survived','True Survived'])

#調整測試資料集以符合模型的輸入格式
#Model Using
import joblib
joblib.dump(lr,'SpaceShipTitanic-LR-20241005.pkl',compress=3)











#final = pd.DataFrame()