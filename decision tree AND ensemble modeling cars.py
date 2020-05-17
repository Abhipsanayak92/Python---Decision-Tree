# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:36:30 2019

@author: user
"""

#%%%
import pandas as pd
import numpy as np
#%%
cars_data=pd.read_csv(r"D:\DATA SCIENCE DOCS\Python docs\6 decision tree python\decision tree and ensemble modeling dataset cars.csv", header=None)
 
cars_data.head()
 
#%%
cars_data.shape
 
#%%
cars_data.columns=["buying", "maint", "doors", "persons","lug_boot","safety","classes"]
 
#%%

cars_data.head()

#%%
cars_data.isnull().sum()

cars_df=pd.DataFrame.copy(cars_data)

colname=cars_df.columns[:]
colname


#%% everything converted to numbers
#labelenoding always done in alphabetical order]
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for x in colname:
    cars_df[x]= le.fit_transform(cars_df[x])
    
 #%%
cars_df.head()
 
 #%%
 X=cars_df.values[:,:-1] 
 Y= cars_df.values[:,-1]
 
 #%%
 
 from sklearn.preprocessing import StandardScaler
 #to scale the data to unit variance ie normalizing data
 scaler=StandardScaler()
 scaler.fit(X)
 X=scaler.transform(X)
 
 #%%
 from sklearn.model_selection import train_test_split
 
 #split the data in to train and test
 
 X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.3, random_state=10)

#%% running decision tree model
 
 #predicting using the decision_Tree_Classifier
 
 from sklearn.tree import DecisionTreeClassifier
 
 model_DecisionTree = DecisionTreeClassifier(random_state=10)
 
 #fit the model on the data and predict the values
 
 model_DecisionTree.fit(X_train, Y_train)
 
 #%%
 
 Y_pred=model_DecisionTree.predict(X_test)
 
 #print(Y_pred)
 
 print(list(zip(Y_test, Y_pred)))  #to merge 2 lists, zip() used
 
 #%%
 
 from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
 
 print(confusion_matrix(Y_test, Y_pred))
 
 print(accuracy_score(Y_test,Y_pred))
 print(classification_report(Y_test, Y_pred))


#%% executing through svm
 
from sklearn.svm import SVC
#model=SVC(kernel='rbf',C=1.0,gamma=0.1)  #here svm gives accuracy 85.54
#svc- support vector classifier
#svr- support vector regression
#from sklearn.linear_model import LogisticRegression
#svc_model=LogisticRegression()

model=SVC(kernel='rbf',C=70.0,gamma=0.1) #here svm gives accuracy 99.42
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(list(Y_pred))


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
 
print(confusion_matrix(Y_test, Y_pred))
 
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test, Y_pred))


#here svm gives accuracy 85.54 without tuning and with tuning it is giving 99.42
#and decision tree gives 99.92 by default
#so Decision tree is suitable for this model

#%% implementing logistics regression

from sklearn.linear_model import LogisticRegression

model=LogisticRegression() 
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(list(Y_pred))


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
 
print(confusion_matrix(Y_test, Y_pred))
 
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test, Y_pred))


####here it gives 69.94 so it fails in multi class classification

#%%
from sklearn import tree
with open(r"D:\DATA SCIENCE DOCS\Python docs\decision tree python\graphiz.txt", "w") as f:  
    f = tree.export_graphviz(model_DecisionTree, feature_names= colname[:-1],out_file=f)
#generate the file and upload the code in webgraphviz.com to plot the decision tree
    
#%% feature importance attribute of decision tree
    print(list(zip(colname,model_DecisionTree.feature_importances_)))
    
    
#%%
#predicting using the Bagging_Classifier
from sklearn.ensemble import ExtraTreesClassifier
#model=(ExtraTreesClassifier(21,random_state=10)) 
#21 is randomly taken for no of bags or decision trees
#the default no of trees it will run =10


#fit the model on the data and predict the values
model=(ExtraTreesClassifier(150,random_state=10))
model=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))    


#%%
#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier
model_RandomForest=RandomForestClassifier(50, random_state=10)
#by default value =10
#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)
Y_pred=model_RandomForest.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


#%%
#predicting using the AdaBoost_Classifier

#default no of trees or estimators= 50
from sklearn.ensemble import AdaBoostClassifier
model_AdaBoost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),random_state=10)
#fit the model on the data and predict the values
model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#%%

#predicting using the Gradient_Boosting_Classifier

#default no of trees or estimators= 100
from sklearn.ensemble import GradientBoostingClassifier
model_GradientBoosting=GradientBoostingClassifier(random_state=10)
#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)
Y_pred=model_GradientBoosting.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#%% ensemble model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

 

# create the sub models
estimators = []
#model1 = LogisticRegression()
#estimators.append(('log', model1))
#if we will run log reg then acc will be low as it is not suitable for multi class classification
model2 = DecisionTreeClassifier(random_state=10)
estimators.append(('cart', model2))
model3 = SVC(kernel="rbf", C=70, gamma= 0.1)
estimators.append(('svm', model3))
print(estimators)

 
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
