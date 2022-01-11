
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


class LogReg:
    """Logistic Regression model for classification takes lengthy amount of time to run """

    def __init__(self, dataFrame):
        self._data = dataFrame
        self._logReg =   LogisticRegression(C=1, max_iter=10000, solver='saga')
    
    def splitTrainTestSet(self, dataFrame):
        """Splits data in a 80/20 split for training/testing based on readmitted column"""
        dataFrame = pd.DataFrame(dataFrame, columns=self._data.columns)
        y = dataFrame['readmitted']
        x = dataFrame.drop(['readmitted'], axis = 1)
        x = dataFrame.iloc[:, 2:dataFrame.shape[1]-1]
      
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=1)

        return x_train, x_test, y_train, y_test

    
    def classify(self,dataFrame):
        """Classify and predict with the model then print the stats for the model(Accuracy, Recall, Precision, F1, Confusion Matrix)"""
       
        x_train, x_test, y_train, y_test = self.splitTrainTestSet(dataFrame)

      
        self._logReg = self._logReg.fit(x_train, y_train)

        y_pred = self._logReg.predict(x_test)

        self.printStats(y_test, y_pred, "Logistic Regression")
        

        

   



    
    def getLearningCurve(self,dataFrame):
        """Builds a learning curve using the fitted model and displays it as a graph in order to evaluate model the colors on the graph indicate standard deviation"""
        

        y = dataFrame['readmitted'].astype(str)
        x = dataFrame.drop(['readmitted'], axis = 1)
        x = dataFrame.iloc[:, 2:dataFrame.shape[1]-1]
        train_sizes = [1, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7914]

        print("Building Learning Curve this will take a while...")

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator= self._logReg,
            X = x,
            y = y, train_sizes = train_sizes, cv = 5,
            scoring = 'neg_mean_squared_error'
           
            
            
        )

        train_scores_mean = np.mean(train_scores,axis = 1)
        train_scores_std = np.std(train_scores, axis=1)
        validation_scores_mean = np.mean(validation_scores,axis = 1)
        validation_scores_std = np.std(validation_scores,axis = 1)

        plt.style.use('seaborn')
        

        plt.plot(train_sizes, abs(train_scores_mean), 'o-', color="r", label="Training score")
        plt.plot(train_sizes, abs(validation_scores_mean), 'o-', color="g", label="Cross-validation score")

        plt.fill_between(train_sizes, abs(train_scores_mean - train_scores_std), abs(train_scores_mean + train_scores_std), alpha=0.1, color="r")
        plt.fill_between(train_sizes, abs(validation_scores_mean - validation_scores_std), abs(validation_scores_mean + validation_scores_std), alpha=0.1, color="g")

        plt.ylabel('Score', fontsize = 14)
        plt.xlabel('Training Set Size', fontsize = 14)
        plt.title("Learning Curve for a Logistic Regression Model", fontsize = 18, y = 1.03)
        plt.legend()
        plt.gca().invert_yaxis()
        
       
        plt.savefig("LogisticRegressionLearningCurve.png")
        plt.show()

        
        
    
        
   


    

    
    def printStats(self,y_test, y_pred, type):
        """Prints the stats of the model to the screen (Accuracy, Precision, Recall, F1, Confusion Matrix)"""
        print(f"Stats for {type} \n")
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred, average="weighted"))
        print("Recall:", metrics.recall_score(y_test, y_pred, average="weighted"))
        print("F1 Score:", metrics.f1_score(y_test, y_pred))
        print("Classification Report:\n",metrics.classification_report(y_test, y_pred))
        print("\n\n Confusion Matrix: ")
        print(confusion_matrix(y_test, y_pred))

