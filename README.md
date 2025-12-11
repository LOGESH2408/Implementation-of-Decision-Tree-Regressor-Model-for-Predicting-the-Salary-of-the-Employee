# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:LOGESHWARAN S

RegisterNumber:25007255  
*/
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
df.info()
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()

x=df[['Position','Level']]
x.head()
y=df['Salary']
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
<img width="489" height="333" alt="Screenshot 2025-12-11 084634" src="https://github.com/user-attachments/assets/8c95f031-e9dc-44a0-9e9a-2f4651a067ea" />

<img width="360" height="226" alt="Screenshot 2025-12-11 084642" src="https://github.com/user-attachments/assets/51948832-eb55-4ff3-8254-c0083f9ad08c" />

<img width="356" height="139" alt="Screenshot 2025-12-11 084701" src="https://github.com/user-attachments/assets/7a12cf0f-14a2-4db4-a08e-a28c8f7f14d8" />

<img width="253" height="37" alt="Screenshot 2025-12-11 084710" src="https://github.com/user-attachments/assets/ff08b0ff-c13a-4b9b-971c-15a2f567ec4c" />

<img width="262" height="40" alt="Screenshot 2025-12-11 084718" src="https://github.com/user-attachments/assets/fe789b32-f4bb-4978-9668-2759f9ab8e74" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
