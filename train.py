import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


#dagshub.init(repo_owner='avi350751', repo_name='mlflow-dagshub', mlflow=True)
#mlflow.set_tracking_uri("https://dagshub.com/avi350751/mlflow-dagshub.mlflow")

#load the dataset
iris = pd.read_csv('https://raw.githubusercontent.com/avi350751/my-datasets/main/iris.csv')
X = iris.iloc[:,0:-1]
y = iris.iloc[:,-1]

#Split the dataset into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Define the parameters of RF model
max_depth = 1
n_estimators = 100


# Apply mlflow

mlflow.set_experiment('iris-rf')

with mlflow.start_run(run_name = 'avi-new-exp1'):

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.set_tag('author','rick')
    mlflow.set_tag('model','random_forest')
   
    #Create a confusion matrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True,fmt ='d',cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    #Save the plot as an artifact
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    #Log the model
    mlflow.sklearn.log_model(rf,"random_forest_model")

    #set tag
    mlflow.set_tag('author','avi')
    mlflow.set_tag('model','random_forest')


    ##logging datasets
    train_df = X_train
    train_df['variety'] = y_train

    test_df = X_test
    test_df['variety'] = y_test

    train_df = mlflow.data.from_pandas(train_df)
    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_df, 'train')
    mlflow.log_input(test_df,'validation')

    print('accuracy is :',accuracy)