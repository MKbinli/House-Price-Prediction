import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def detectNonStdFeatures(df):
    featureNames=[]
    columnNames=df.columns
    func=lambda x: True if x>1 else False
    for feat in columnNames:
        if(df[feat].apply(func).any()):
            featureNames.append(feat)
    return featureNames

selectedFeatures=['MSZoning', 'Neighborhood', 'OverallQual', 'OverallCond',
       'BsmtQual', 'BsmtFinType1', 'CentralAir', 'GarageFinish',
       'LotArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'TotRmsAbvGrd', 'Fireplaces','YrSold','YearBuilt','YearRemodAdd']#'NeighborhoodGroup4,'YearBuildDif'

dfTrain=pd.read_csv("train.csv")
dfTest=pd.read_csv("test.csv")

Xtrain=dfTrain[selectedFeatures].copy()
Xtest=dfTest[selectedFeatures].copy()

yTrain=dfTrain['SalePrice'].copy()


Xtrain.reset_index(drop=True,inplace=True)
Xtest.reset_index(drop=True,inplace=True)
yTrain.reset_index(drop=True,inplace=True)

Xtrain.isna().sum()
Xtest.isna().sum()
"""
BsmtFinSF1       1
TotalBsmtSF      1
BsmtFullBath     2
"""

medianImpute=Xtrain["BsmtFinSF1"].median()
Xtest["BsmtFinSF1"]=Xtest["BsmtFinSF1"].fillna(medianImpute)
medianImpute=Xtrain["TotalBsmtSF"].median()
Xtest["TotalBsmtSF"]=Xtest["TotalBsmtSF"].fillna(medianImpute)
medianImpute=Xtrain["BsmtFullBath"].median()
Xtest["BsmtFullBath"]=Xtest["BsmtFullBath"].fillna(medianImpute)

Xtrain['MSZoning']=Xtrain["MSZoning"].fillna("Missing")
Xtest['MSZoning']=Xtest["MSZoning"].fillna("Missing")
Xtrain['MSZoning']=Xtrain['MSZoning'].map({"RL":1,'RM':0,'C (all)':0, 'FV':0, 'RH':0,'A':0,'I':0,'RP':0,"Missing":0})
Xtest['MSZoning']=Xtest['MSZoning'].map({"RL":1,'RM':0,'C (all)':0, 'FV':0, 'RH':0,'A':0,'I':0,'RP':0,"Missing":0})

mapperFunc=lambda x: 1 if x=='Blmngtn' or x=='CollgCr' or x=='Crawfor' or x=='ClearCr' or x=='Somerst' else 0

Xtrain['Neighborhood']=Xtrain['Neighborhood'].apply(mapperFunc)
Xtest['Neighborhood']=Xtest['Neighborhood'].apply(mapperFunc)

Xtrain["BsmtQual"]=Xtrain["BsmtQual"].fillna("Missing")
Xtest["BsmtQual"]=Xtest["BsmtQual"].fillna("Missing")
Xtrain["BsmtQual"]=Xtrain["BsmtQual"].map({"Missing":1,'Po':1, 'Fa':1, 'TA':1, 'Gd':2, 'Ex':3})
Xtest["BsmtQual"]=Xtest["BsmtQual"].map({"Missing":1,'Po':1, 'Fa':1, 'TA':1, 'Gd':2, 'Ex':3})


Xtrain["BsmtFinType1"]=Xtrain["BsmtFinType1"].fillna("Missing")
Xtest["BsmtFinType1"]=Xtest["BsmtFinType1"].fillna("Missing")
Xtrain["BsmtFinType1"]=Xtrain["BsmtFinType1"].map({'Missing':1, 'NA':1, 'LwQ':1, 'Rec':1, 'BLQ':1, 'Unf':2,'ALQ':3,'GLQ':4})
Xtest["BsmtFinType1"]=Xtest["BsmtFinType1"].map({'Missing':1, 'NA':1, 'LwQ':1, 'Rec':1, 'BLQ':1, 'Unf':2,'ALQ':3,'GLQ':4})

Xtrain["CentralAir"]=Xtrain["CentralAir"].map({'Y':1, 'N':0})
Xtest["CentralAir"]=Xtest["CentralAir"].map({'Y':1, 'N':0})

Xtrain["GarageFinish"]=Xtrain["GarageFinish"].fillna("Missing")
Xtest["GarageFinish"]=Xtest["GarageFinish"].fillna("Missing")
Xtrain["GarageFinish"]=Xtrain["GarageFinish"].map({'Missing':1,'Unf':2, 'RFn':3, 'Fin':4})
Xtest["GarageFinish"]=Xtest["GarageFinish"].map({'Missing':1,'Unf':2, 'RFn':3, 'Fin':4})


columnNames=['LotArea','TotRmsAbvGrd']
transformer = FunctionTransformer(np.log, validate=True)
tempData  = transformer.transform(Xtrain[columnNames])
tempData  = pd.DataFrame(tempData , columns =columnNames)
Xtrain[columnNames]=tempData[columnNames].copy()
tempData = transformer.transform(Xtest[columnNames])
tempData  = pd.DataFrame(tempData , columns =columnNames)
Xtest[columnNames]=tempData[columnNames].copy()

columnNamesGr2=['1stFlrSF','2ndFlrSF','TotalBsmtSF','BsmtFinSF1']
transformer = FunctionTransformer(lambda x: x**(1/2), validate=True)
tempData  = transformer.transform(Xtrain[columnNamesGr2])
tempData  = pd.DataFrame(tempData , columns =columnNamesGr2)
Xtrain[columnNamesGr2]=tempData[columnNamesGr2].copy()
tempData = transformer.transform(Xtest[columnNamesGr2])
tempData  = pd.DataFrame(tempData , columns =columnNamesGr2)
Xtest[columnNamesGr2]=tempData[columnNamesGr2].copy()

funcBsmt=lambda x: 0 if x==0 else 1
Xtrain["BsmtFullBath"]=Xtrain["BsmtFullBath"].apply(funcBsmt)
Xtest["BsmtFullBath"]=Xtest["BsmtFullBath"].apply(funcBsmt)

funcHalf=lambda x: 0 if x==0 else 1
Xtrain['HalfBath']=Xtrain['HalfBath'].apply(funcHalf)
Xtest['HalfBath']=Xtest['HalfBath'].apply(funcHalf)

discFeatures=['FullBath', 'BedroomAbvGr', 'Fireplaces']
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
disc.fit(Xtrain[discFeatures])
trainTransformed = disc.transform(Xtrain[discFeatures])
trainTransformed = pd.DataFrame(trainTransformed , columns = discFeatures)

testTransformed = disc.transform(Xtest[discFeatures])
testTransformed = pd.DataFrame(testTransformed , columns = discFeatures)

Xtrain[discFeatures]=trainTransformed[discFeatures].copy()
Xtest[discFeatures]=testTransformed[discFeatures].copy()

Xtrain["YearBuildDif"]=Xtrain["YrSold"]-Xtrain[["YearBuilt","YearRemodAdd"]].max(axis=1)
Xtest["YearBuildDif"]=Xtest["YrSold"]-Xtest[["YearBuilt","YearRemodAdd"]].max(axis=1)
Xtrain.drop(columns=["YearBuilt","YearRemodAdd","YrSold"],inplace=True)
Xtest.drop(columns=["YearBuilt","YearRemodAdd","YrSold"],inplace=True)

numericalScalable=detectNonStdFeatures(Xtrain)

scaler = StandardScaler()
scaler.fit(Xtrain[numericalScalable])

scaledTrain = scaler.transform(Xtrain[numericalScalable])
scaledTest = scaler.transform(Xtest[numericalScalable])

Xtrain[numericalScalable] = pd.DataFrame(scaledTrain, columns=numericalScalable)
Xtest[numericalScalable] = pd.DataFrame(scaledTest, columns=numericalScalable)

def buildDataset(dfTrain,dfTest):
    selectedFeatures=['MSZoning', 'Neighborhood', 'OverallQual', 'OverallCond',
       'BsmtQual', 'BsmtFinType1', 'CentralAir', 'GarageFinish',
       'LotArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'TotRmsAbvGrd', 'Fireplaces','YrSold','YearBuilt','YearRemodAdd']#'NeighborhoodGroup4,'YearBuildDif'

    Xtrain=dfTrain[selectedFeatures].copy()
    Xtest=dfTest[selectedFeatures].copy()

    yTrain=dfTrain['SalePrice'].copy()


    Xtrain.reset_index(drop=True,inplace=True)
    Xtest.reset_index(drop=True,inplace=True)
    yTrain.reset_index(drop=True,inplace=True)

    print(Xtrain.isna().sum())
    print(Xtest.isna().sum())

    medianImpute=Xtrain["BsmtFinSF1"].median()
    Xtest["BsmtFinSF1"]=Xtest["BsmtFinSF1"].fillna(medianImpute)
    medianImpute=Xtrain["TotalBsmtSF"].median()
    Xtest["TotalBsmtSF"]=Xtest["TotalBsmtSF"].fillna(medianImpute)
    medianImpute=Xtrain["BsmtFullBath"].median()
    Xtest["BsmtFullBath"]=Xtest["BsmtFullBath"].fillna(medianImpute)

    Xtrain['MSZoning']=Xtrain["MSZoning"].fillna("Missing")
    Xtest['MSZoning']=Xtest["MSZoning"].fillna("Missing")
    Xtrain['MSZoning']=Xtrain['MSZoning'].map({"RL":1,'RM':0,'C (all)':0, 'FV':0, 'RH':0,'A':0,'I':0,'RP':0,"Missing":0})
    Xtest['MSZoning']=Xtest['MSZoning'].map({"RL":1,'RM':0,'C (all)':0, 'FV':0, 'RH':0,'A':0,'I':0,'RP':0,"Missing":0})

    mapperFunc=lambda x: 1 if x=='Blmngtn' or x=='CollgCr' or x=='Crawfor' or x=='ClearCr' or x=='Somerst' else 0

    Xtrain['Neighborhood']=Xtrain['Neighborhood'].apply(mapperFunc)
    Xtest['Neighborhood']=Xtest['Neighborhood'].apply(mapperFunc)

    Xtrain["BsmtQual"]=Xtrain["BsmtQual"].fillna("Missing")
    Xtest["BsmtQual"]=Xtest["BsmtQual"].fillna("Missing")
    Xtrain["BsmtQual"]=Xtrain["BsmtQual"].map({"Missing":1,'Po':1, 'Fa':1, 'TA':1, 'Gd':2, 'Ex':3})
    Xtest["BsmtQual"]=Xtest["BsmtQual"].map({"Missing":1,'Po':1, 'Fa':1, 'TA':1, 'Gd':2, 'Ex':3})


    Xtrain["BsmtFinType1"]=Xtrain["BsmtFinType1"].fillna("Missing")
    Xtest["BsmtFinType1"]=Xtest["BsmtFinType1"].fillna("Missing")
    Xtrain["BsmtFinType1"]=Xtrain["BsmtFinType1"].map({'Missing':1, 'NA':1, 'LwQ':1, 'Rec':1, 'BLQ':1, 'Unf':2,'ALQ':3,'GLQ':4})
    Xtest["BsmtFinType1"]=Xtest["BsmtFinType1"].map({'Missing':1, 'NA':1, 'LwQ':1, 'Rec':1, 'BLQ':1, 'Unf':2,'ALQ':3,'GLQ':4})

    Xtrain["CentralAir"]=Xtrain["CentralAir"].map({'Y':1, 'N':0})
    Xtest["CentralAir"]=Xtest["CentralAir"].map({'Y':1, 'N':0})

    Xtrain["GarageFinish"]=Xtrain["GarageFinish"].fillna("Missing")
    Xtest["GarageFinish"]=Xtest["GarageFinish"].fillna("Missing")
    Xtrain["GarageFinish"]=Xtrain["GarageFinish"].map({'Missing':1,'Unf':2, 'RFn':3, 'Fin':4})
    Xtest["GarageFinish"]=Xtest["GarageFinish"].map({'Missing':1,'Unf':2, 'RFn':3, 'Fin':4})


    columnNames=['LotArea','TotRmsAbvGrd']
    transformer = FunctionTransformer(np.log, validate=True)
    tempData  = transformer.transform(Xtrain[columnNames])
    tempData  = pd.DataFrame(tempData , columns =columnNames)
    Xtrain[columnNames]=tempData[columnNames].copy()
    tempData = transformer.transform(Xtest[columnNames])
    tempData  = pd.DataFrame(tempData , columns =columnNames)
    Xtest[columnNames]=tempData[columnNames].copy()

    columnNamesGr2=['1stFlrSF','2ndFlrSF','TotalBsmtSF','BsmtFinSF1']
    transformer = FunctionTransformer(lambda x: x**(1/2), validate=True)
    tempData  = transformer.transform(Xtrain[columnNamesGr2])
    tempData  = pd.DataFrame(tempData , columns =columnNamesGr2)
    Xtrain[columnNamesGr2]=tempData[columnNamesGr2].copy()
    tempData = transformer.transform(Xtest[columnNamesGr2])
    tempData  = pd.DataFrame(tempData , columns =columnNamesGr2)
    Xtest[columnNamesGr2]=tempData[columnNamesGr2].copy()

    funcBsmt=lambda x: 0 if x==0 else 1
    Xtrain["BsmtFullBath"]=Xtrain["BsmtFullBath"].apply(funcBsmt)
    Xtest["BsmtFullBath"]=Xtest["BsmtFullBath"].apply(funcBsmt)

    funcHalf=lambda x: 0 if x==0 else 1
    Xtrain['HalfBath']=Xtrain['HalfBath'].apply(funcHalf)
    Xtest['HalfBath']=Xtest['HalfBath'].apply(funcHalf)

    discFeatures=['FullBath', 'BedroomAbvGr', 'Fireplaces']
    disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    disc.fit(Xtrain[discFeatures])
    trainTransformed = disc.transform(Xtrain[discFeatures])
    trainTransformed = pd.DataFrame(trainTransformed , columns = discFeatures)

    testTransformed = disc.transform(Xtest[discFeatures])
    testTransformed = pd.DataFrame(testTransformed , columns = discFeatures)

    Xtrain[discFeatures]=trainTransformed[discFeatures].copy()
    Xtest[discFeatures]=testTransformed[discFeatures].copy()

    Xtrain["YearBuildDif"]=Xtrain["YrSold"]-Xtrain[["YearBuilt","YearRemodAdd"]].max(axis=1)
    Xtest["YearBuildDif"]=Xtest["YrSold"]-Xtest[["YearBuilt","YearRemodAdd"]].max(axis=1)
    Xtrain.drop(columns=["YearBuilt","YearRemodAdd","YrSold"],inplace=True)
    Xtest.drop(columns=["YearBuilt","YearRemodAdd","YrSold"],inplace=True)

    numericalScalable=detectNonStdFeatures(Xtrain)

    scaler = StandardScaler()
    scaler.fit(Xtrain[numericalScalable])

    scaledTrain = scaler.transform(Xtrain[numericalScalable])
    scaledTest = scaler.transform(Xtest[numericalScalable])

    Xtrain[numericalScalable] = pd.DataFrame(scaledTrain, columns=numericalScalable)
    Xtest[numericalScalable] = pd.DataFrame(scaledTest, columns=numericalScalable)
    return Xtrain, Xtest, yTrain
