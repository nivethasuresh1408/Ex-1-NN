<H3>ENTER YOUR NAME : NIVETHA SURESH S</H3>
<H3>ENTER YOUR REGISTER NO : 212223040137</H3>
<H3>EX. NO . 1</H3>
<H3>DATE : 10 / 03 / 2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

*Kaggle :*
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

*Data Preprocessing:*

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

*Need of Data Preprocessing :*

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
import libraries
```
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
read the dataset
```
df=pd.read_csv("Churn_Modelling.csv")
df.head()
df.tail()
df.columns
```
check the missing data 
```
df.isnull().sum()
df.duplicated()
```
assigning y
```
y = df.iloc[:, -1].values
print(y)
```
check for duplicates 
```
df.duplicated()
```
check for outliers
```
df.describe()
```
droping string values data from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
Checking datasets after dropping string values data from dataset
```
data.head()
```
Normalize the dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
Split the dataset
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
Training and testing model
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:
Data checking

![data checking](https://github.com/user-attachments/assets/3a43853a-fbc9-4dd5-91b4-ee1beebaf72c)

Duplicates Identification

![duplicates identification](https://github.com/user-attachments/assets/e4f93cde-80c7-417b-83b1-ff72d386e41f)

Values of 'Y'

![values of y](https://github.com/user-attachments/assets/3a7dbd1d-5023-4024-b2d3-4d04191688ef)

Outliers

![Outliers](https://github.com/user-attachments/assets/2e54c620-fb07-4a95-a638-a147d729ab80)

Checking datasets after dropping string values data from dataset

![image](https://github.com/user-attachments/assets/6573d3d4-a3c2-4b69-ae6c-60392e8ad1d8)

Normalize the dataset

![normalize](https://github.com/user-attachments/assets/bfa826f7-cfdd-4715-97bb-1311652ca845)

Split the dataset

![spilt](https://github.com/user-attachments/assets/68d9d037-7dbf-4b6c-a2ad-86b78b628bc5)

Training the Model

![training the model](https://github.com/user-attachments/assets/b764f7a5-a581-424d-a38d-bc4836278e29)

Testing the Model

![testing ](https://github.com/user-attachments/assets/053f9425-d1f2-40c0-ae13-3e27e2118753)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
