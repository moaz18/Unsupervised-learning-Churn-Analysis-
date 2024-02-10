# TELECOM CHURN ANALYSIS - SUPERVISED MACHINE LEARNING

#Lets import the relevant library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#lets generate our own Dataset

np.random.seed(42) #----> Used to generate same random numbers again and again and simplifies algorithm testing process

telecom_data = pd.DataFrame({
    'Monthly_charges':np.random.randint(30,100,500),
    'Total_charges': np.random.uniform(100,5000,500),
    'Tenure':np.random.uniform(1,72,500),
    'Churn':np.random.choice([0,1],500,p=[0.8,0.2])
})

# Exploratory Data Analysis (EDA)

#lets do some descriptive stats
telecom_data.describe() # ----> Used for all descriptive stats related information 
telecom_data.info()
telecom_data.isnull().sum()
telecom_data.columns # ----> Shows all the columns ie features in your dataset


# Plotting the data

plt.boxplot(telecom_data) #--> for outliers detection
plt.xlabel('features') # ----> setting up the labels on x axis
plt.ylabel('Churn Analysis') 
plt.title('Telecom Churn Analysis') # ---> setting up the title
plt.show()

# Feature Selection

features = telecom_data[['Monthly_charges','Total_charges','Tenure']]

target = telecom_data['Churn']


# lets split the data into train and test

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.8,train_size=0.2,random_state=42) 

# lets select the model according to our problem

model_selection = RandomForestClassifier(n_estimators=100,random_state=42) # ---> In this case we choose Random Forest Classifier because we are dealing with Unsupervised learning

model_selection.fit(X_train,y_train)

# Model Prediction

y_pred = model_selection.predict(X_test)

#print(y_pred)

accuracy = accuracy_score(y_pred,y_test)

#print(accuracy)

confusion_mat = confusion_matrix(y_pred,y_test)

#print(confusion_mat)

classific_report = classification_report(y_pred,y_test)

print(classific_report)

# Heat Map
correlation_matrix = telecom_data.corr()

#sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')

#plt.show()

#visualize the distribution of target variable "CHURN"

#sns.countplot(x='Churn',#data=telecom_data)
#plt.show()

