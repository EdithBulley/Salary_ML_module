#importing labraries in  machine learning to perform maths
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Creating a variable to store a data
Dataset = pd.read_csv("Salary_Data.csv")

#transforming the dependent and independent variable (x)

x = Dataset.iloc[0:30,0:1].values
y = Dataset.iloc[0:30,0:2].values

#splitting the dataset into train data and test data 
from sklearn.model_selection import train_test_split

#creating variable to store X_train and Y_train
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)


#This is to show how to train a module
from sklearn.linear_model import LinearRegression

#create a variable to assign a module
Dataset_module = LinearRegression()

 #This is to show how to train the module
 Dataset_module.fit(x_train,y_train)

#making a prediction
prediction_result = Dataset_module.predict(x_test)
prediction_result
  