#Project for machine learning
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

def load_robot_data():
     ifile = open("phpVeNa5j.csv", "rb")
     return pd.read_csv(ifile)

def train_model(X_train, y_train_x, predictor, prediction_string):
   sgd_clf=SGDClassifier(random_state=42, max_iter=5000)
   sgd_clf.fit(X_train, y_train_x)
   print(prediction_string, sgd_clf.predict(predictor.reshape(1,-1)))
   print("Cross Val Score: ", cross_val_score(sgd_clf, X_train, y_train_x, cv=3, scoring="accuracy"))

   #Confusion matrix
   y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_x, cv=3)
   print("Confusion Matrix: ", confusion_matrix(y_train_x, y_train_pred))

   y_scores = cross_val_predict(sgd_clf, X_train, y_train_x, cv=3, method="decision_function")
    
   precisions, recalls, thresholds = precision_recall_curve(y_train_x, y_scores)

   plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
   plt.show()

   #making predictions
   y_train_pred_90 = (y_scores > 4500)
   print("Precision Score: ", precision_score(y_train_x, y_train_pred_90))
   print("Recall Scores: ", recall_score(y_train_x, y_train_pred_90))

   #Roc curve
   fpr, tpr, thresholds = roc_curve(y_train_x, y_scores)
   plot_roc_curve(fpr, tpr)
   plt.show()

   print("ROC Score: ", roc_auc_score(y_train_x, y_scores))


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0,1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



robot = load_robot_data()
# train_set, test_set = train_test_split(robot, test_size=0.2, random_state=42)

#training set
y = robot["Class"]
X = robot[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24" ]]
y= y.to_numpy()
X= X.to_numpy()

#What we want to predict 

leftTurnPredict = X[3] #Left Turn
sharpRightTurnPredict = X[1680] #Sharp Right
slightRightTurnPredict = X[20] #Slight Right
forwardPredict = X[67] #Forward

X_train, X_test, y_train, y_test = X[:3500], X[3500:], y[:3500], y[3500:]
#Shuffle the data
shuffle_index = np.random.permutation(3500)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


#testing binary classsifcation of 4(Slight left turn)
y_train_4 = (y_train==4)
y_test_4 = (y_test==4)
#testing binary classsifcation of 3(Sharp Right turn)
y_train_3 = (y_train==3)
y_test_3 = (y_test==3)
#testing binary classsifcation of 2(Slight Right turn)
y_train_2 = (y_train==2)
y_test_2 = (y_test==2)
#testing binary classsifcation of 1(Forward)
y_train_1 = (y_train==1)
y_test_1 = (y_test==1)

train_model(X_train, y_train_4, leftTurnPredict, "Prediction for Left Turn: ")
train_model(X_train, y_train_3, sharpRightTurnPredict, "Prediction for Sharp Right Turn: ")
train_model(X_train, y_train_2, slightRightTurnPredict, "Prediction for Slight Right Turn: ")
train_model(X_train, y_train_1, forwardPredict, "Prediction for moving forward: ")




