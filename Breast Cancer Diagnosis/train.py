import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# load the dataset
df =pd.read_csv(" data/data (1).csv")
df1 =df.drop(['id','Unnamed: 32'],axis=1)
print(df.head())
print(df.info())

X = df1.drop('diagnosis',axis=1)
Y = df1['diagnosis']
print(X.info())
print(Y.value_counts())

# split the dataset into training and testing sets
x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=0.3,random_state=42)

# define hyperparameter search spases for training
param_grid ={
    'max_depth':[3,6,10,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,3,5],
    'criterion':["gini","entropy"],
    'max_features':['sqrt', 'log2', None]
}

# initializing gridsearchCV

grid_search = GridSearchCV(DecisionTreeClassifier(),param_grid,cv = 5,scoring ='accuracy',n_jobs=-1)

# training the model
print('Training the model with different parameters')
grid_search.fit(x_train,y_train)

# getting the best parameters
print('best parameters:',grid_search.best_params_)

best_model = grid_search.best_estimator_

# training the model with best parameters
best_model.fit(x_train,y_train)

# prediction on test set
predictions = best_model.predict(x_test)

# calculate accuracy
accuracy = accuracy_score(y_test,predictions)
print(f'Accuracy of the model: {accuracy*100}')

# saving the trained model
joblib.dump(best_model,'model/model.pkl')








