import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

df=pd.read_csv('IceCream.csv')

df

df.describe()

df.info()

df.isnull().sum()

X=df.drop('Revenue',axis=1)
y=df['Revenue']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(df['Temperature'])

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_test

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
model=lr.predict([[26]])
print(model)

with open('model.pkl', 'wb') as f:
    pickle.dump(lr, f)