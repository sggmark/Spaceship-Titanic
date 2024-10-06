import numpy as np
import pandas as pd
import joblib

model_pretrained = joblib.load('SpaceShipTitanic-LR-20241005.pkl') #Load the model

#for submission
df_test = pd.read_csv("test.csv")
df_test.head()
df_test.drop(["Name"], axis=1, inplace=True)
df_test['CryoSleep'].fillna(df_test['CryoSleep'].value_counts().idxmax(),inplace=True)
df_test.loc[df_test['CryoSleep'] == True, 'RoomService'] = df_test.loc[df_test['CryoSleep'] == True, 'RoomService'].fillna(0)
df_test['RoomService'].fillna(df_test['RoomService'].median(),inplace=True)
df_test.loc[df_test['CryoSleep'] == True, 'FoodCourt'] = df_test.loc[df_test['CryoSleep'] == True, 'FoodCourt'].fillna(0)
df_test['FoodCourt'].fillna(df_test['FoodCourt'].median(),inplace=True)
df_test.loc[df_test['CryoSleep'] == True, 'Spa'] = df_test.loc[df_test['CryoSleep'] == True, 'Spa'].fillna(0)
df_test['Spa'].fillna(df_test['Spa'].median(),inplace=True)
df_test.loc[df_test['CryoSleep'] == True, 'VRDeck'] = df_test.loc[df_test['CryoSleep'] == True, 'VRDeck'].fillna(0)
df_test['VRDeck'].fillna(df_test['VRDeck'].median(),inplace=True)
df_test.loc[df_test['CryoSleep'] == True, 'ShoppingMall'] = df_test.loc[df_test['CryoSleep'] == True, 'ShoppingMall'].fillna(0)
df_test['ShoppingMall'].fillna(df_test['ShoppingMall'].median(),inplace=True)
df_test['HomePlanet'].fillna(df_test['HomePlanet'].value_counts().idxmax(),inplace=True)
df_test = pd.get_dummies(data=df_test,dtype=int,columns=['HomePlanet'])
df_test[['Cabin_lvl', 'Cabin_num', 'Cabin_ps']] = df_test['Cabin'].str.split('/', expand=True)
df_test.drop("Cabin",axis=1,inplace=True)
df_test = pd.get_dummies(data=df_test,dtype=int,columns=['Cabin_ps'])
df_test.drop(['Cabin_ps_S'],axis=1,inplace=True)
df_test['Cabin_ps_P'].fillna(df_test['Cabin_ps_P'].value_counts().idxmax(),inplace=True)
df_test['Cabin_num'].fillna(df_test['Cabin_num'].value_counts().idxmax(),inplace=True)
df_test['Cabin_lvl'] = df_test['Cabin_lvl'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8})
df_test['Cabin_lvl'].fillna(df_test['Cabin_lvl'].value_counts().idxmax(),inplace=True)
df_test['Age'].fillna(df_test['Age'].median(),inplace=True)
df_test['VIP'].fillna(df_test['VIP'].value_counts().idxmax(),inplace=True)
df_test = pd.get_dummies(data=df_test,dtype=int,columns=['VIP'])
df_test['Destination'] = df_test['Destination'].map({'TRAPPIST-1e':1, '55 Cancri e':2, 'PSO J318.5-22':3})
df_test['Destination'].fillna(df_test['Destination'].value_counts().idxmax(),inplace=True)
df_test['CryoSleep'] = df_test['CryoSleep'].map({True:1,False:0})

passenger_id = df_test["PassengerId"]
df_test.drop(["PassengerId"], axis=1, inplace=True)

predictions2 = model_pretrained.predict(df_test)

#prepare for submission
forSubmissionDF = pd.DataFrame(
    {
        "PassengerId":passenger_id, 
        "Transported":predictions2
        }
        )

forSubmissionDF.to_csv("for_submission_20241005.csv", index=False)
