#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def cleanUp(data):

    #Imputing data    
    data["Age"].fillna(round(data["Age"].mean()),inplace=True)                              #replacing missing values for integer columns
    data["Year of Record"].fillna(data["Year of Record"].mode()[0],inplace=True)
    data["Size of City"].fillna(round(data["Size of City"].mean()),inplace=True)    
    
    data["Gender"].fillna(data["Gender"].mode()[0],inplace=True)                            #replacing missing values for string columns
    data["Country"].fillna(data["Country"].mode()[0],inplace=True)
    data["University Degree"].fillna(data["University Degree"].mode()[0],inplace=True)
    data["Hair Color"].fillna(data["Hair Color"].mode()[0],inplace=True)    
    data[['Profession']]=data[['Profession']].fillna(value='9999')                          #replacing profession missing values with 9999
    
    return data

def getDataDummies(data):
    #one-hot encoding on data
    data=pd.get_dummies(data,columns=['Profession','Year of Record','Gender','Country','University Degree','Wears Glasses','Hair Color'])
    return data

def equalizeColTrain(data1,data2):
                                                                                            #identifying columns that are different in both data sets
    datadiff=data1[data1.columns.difference(data2.columns)]                                 #identifying columns missing in prediction data
    for item in datadiff.columns:
            data2[item]=0
    
    datadiff=data2[data2.columns.difference(data1.columns)]                                 #identifying columns missing in training data
    for item in datadiff.columns:
            data1[item]=0
            
    data2=data2[data1.columns]                                                              #making sure the series of columns in training and prediction data are same        
    
    return data1,data2

def normaliseData(data,feature):
       
  max_value = data[feature].max()                                                           #using min-max scaling for normalisation  
  min_value = data[feature].min()
  data[feature] = (data[feature] - min_value) / (max_value - min_value)
  return data,max_value,min_value

def denormaliseData(data,feature,max_value,min_value):
    data[feature]=data[feature]*(max_value-min_value)+min_value                            #denormalising function
    return data
  
def removeRows(data):

    #outlierCity = detect_outlier(data['Size of City'])                                     #outlier Identification and removal
    #data=data[~data["Size of City"].isin(outlierCity)]
    outlierInc = detect_outlier(data['Income in EUR'])
    data=data[~data["Income in EUR"].isin(outlierInc)]
    #outlierAge = detect_outlier(data['Age'])
    #data=data[~data["Age"].isin(outlierAge)]
    #outlierHt = detect_outlier(data['Body Height [cm]'])
    #data=data[~data["Body Height [cm]"].isin(outlierHt)]
    
    print("\tRemoving rows with negatives income..")                                        #removing rows with negative income
    data = data[(data['Income in EUR']>=0)]
    #print("\tlength: "+str(len(data)))
    
    return data

def detect_outlier(data):
    
    threshold=3
    mean_1 = np.mean(data)
    std_1 =np.std(data)
    outliers=[]
    for y in data:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

def main():

    print("\n#############################Preprocessing#############################\n")
    print("Importing Training data set...")
    data=pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")         #loading training data set
    print("Loading Prediction data Set...")
    dataPred=pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")      #loading prediction data set
    print("Cleaning up...")
    data=cleanUp(data)                                                                      #imputing training data set
    dataPred=cleanUp(dataPred)                                                              #imputing prediction data set
    print("Removing rows...")
    data=removeRows(data)                                                                   #removing irrelevant rows on training data

    print("Normalising...")
    data,maxSizeCityT,minSizeCityT=normaliseData(data,'Size of City')                       #scaling 'Size of City' column in training data set
    data['Income in EUR']=np.log(data['Income in EUR'])                                     #log transformation of Income column
    dataPred,maxSizeCityP,minSizeCityP=normaliseData(dataPred,'Size of City')               #scaling 'Size of City' column in prediction data set
    print("Getting Dummies...")
    data=getDataDummies(data)                                                               #one-hot encoding on training data
    dataPred=getDataDummies(dataPred)                                                       #one-hot encoding on prediction data
    
    print("Equalizing columns...")
    data,dataPred=equalizeColTrain(data,dataPred)                                           #equalizing columns in training and prediction data sets
    
    print ('\tColumns in training :'+str(len(data.columns)))
    print ('\tColumns in prediction :'+str(len(dataPred.columns)))

    train_X=data[data.columns.difference(['Income in EUR','Income','Instance'])]            #setting training features
    train_y=data['Income in EUR']                                                           #setting up training label
    pred_X=dataPred[dataPred.columns.difference(['Income in EUR','Income','Instance'])]     #setting up prediction features
    
    print("Splitting into training and validation data")
    X,X_test,y,y_test = train_test_split(train_X,train_y,train_size=0.70,random_state=42)   #splitting training and validation 70-30
    print('Splitting done..')
    
    print("\n#############################Training#############################\n")
    
    regressor = LinearRegression()                                                          #setting up Linear Regression model
    regressor.fit(X,y)                                                                      #fitting features and labels
    y_pred = regressor.predict(X_test)                                                      #predicting on validation features
    
    rms = sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))                          #evaluating RMSE on validation set
    #rms = sqrt(mean_squared_error(y_test, y_pred))
    print("Error: "+str(rms))   
    
    print("\n#############################Predicting#############################\n")
    y_pred=regressor.predict(pred_X)                                                        #using model to predict Income for prediction feature data
    predinc=pd.DataFrame(np.exp(y_pred))                                                    #exponentiating log transformed predictions and storing in data frame
    #predinc=pd.DataFrame(y_pred)
    predinc.to_csv('finalpreds2.csv')                                                       #exporting predictions to CSVs
    print("Predictions stored as CSV...")
    
    print("\n#############################END#############################\n")
    
if __name__ == '__main__':
    main()
    