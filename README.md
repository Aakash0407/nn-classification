# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS
- STEP 1: Import the packages and reading the dataset.
- STEP 2: Preprocessing and spliting the data.
- STEP 3: Creating a Deep Learning model with appropriate layers of depth.
- STEP 4: Plotting Training Loss, Validation Loss Vs Iteration Plot.
- STEP 5: Predicting the with Sample values.

## PROGRAM

### Name: Aakash P
### Register Number: 212222110001
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
df=pd.read_csv("customers.csv")
df.head()
df.info()
df.isnull().sum()
df=df.drop(['ID','Var_1'],axis=1)
df=df.dropna(axis=0)
for i in ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Segmentation']:
    print(i,":",list(df[i].unique()))
Clist=[['Healthcare','Engineer','Lawyer','Artist','Doctor','Homemaker','Entertainment','Marketing',
        'Executive'],['Male', 'Female'],['No', 'Yes'],['No', 'Yes'],['Low', 'Average', 'High']]
enc = OrdinalEncoder(categories=Clist)
df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]
    =enc.fit_transform(df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])
le = LabelEncoder()
df['Segmentation'] = le.fit_transform(df['Segmentation'])
scaler=MinMaxScaler()
df[['Age']]=scaler.fit_transform(df[['Age']])
X=df.iloc[:,:-1]
Y=df[['Segmentation']]
ohe=OneHotEncoder()
Y=ohe.fit_transform(Y).toarray()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.33,random_state=42)
model=Sequential([Dense(6,activation='relu',input_shape=[8]),Dense(10,activation='relu'),
                  Dense(10,activation='relu'),Dense(4,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy' ,metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=2000,batch_size=32,validation_data=(xtest,ytest))
metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
ypred = np.argmax(model.predict(xtest), axis=1)
ytrue = np.argmax(ytest,axis=1)
print(confusion_matrix(ytrue,ypred))
print(classification_report(ytrue,ypred))
x_single_prediction = np.argmax(model.predict(X[3:4]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```
### Output:
##### Dataset Information
**df.head()** 
![2 1](https://github.com/user-attachments/assets/b5f3f020-4584-4506-850f-780732a2d4ca)


**df.info()** 
![2 2](https://github.com/user-attachments/assets/71cc8e2a-13fd-4cbd-9b37-752c6b9403cd)

**df.isnull().sum()**
![2 3](https://github.com/user-attachments/assets/602fdc37-6a49-42c6-976b-6ddd4fc2f6b5)

### Training Loss, Validation Loss Vs Iteration Plot
![2 4](https://github.com/user-attachments/assets/c106ef82-2b66-4241-9b16-9a2c7e459b93)

### Classification Report
![2 5](https://github.com/user-attachments/assets/1359db93-cfeb-4ada-8376-b8c75b58d6e1)

### Confusion Matrix
![2 6](https://github.com/user-attachments/assets/74b8683c-c519-4423-9252-4ff8528e1114)

### New Sample Data Prediction
![2 7](https://github.com/user-attachments/assets/c69353c5-2a68-47cd-8701-d0ff0531b92d)

## RESULT
A neural network classification model is developed for the given dataset.
