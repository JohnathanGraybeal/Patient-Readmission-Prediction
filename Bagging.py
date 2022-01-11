from sklearn.ensemble import BaggingClassifier
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
class Bagging:
    """Bagging classifier to attempt to get better accuracy by combining all fitted models found in methods parameter"""

    def __init__(self, dataFrame, methods):
        self._data = dataFrame
        self._methods = methods
        

    def splitTrainTestSet(self, dataFrame):
        """Splits data in a 80/20 split for training/testing based on readmitted column"""
        dataFrame = pd.DataFrame(dataFrame, columns=self._data.columns)
        X = dataFrame.drop(['readmitted'], axis=1)
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]-1]
        y = dataFrame['readmitted']
        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

        return x_train, x_test, y_train, y_test

    

    def classify(self, dataFrame):
        """Classify and predict with the model then print the stats for the model(Accuracy, Recall, Precision, F1, Confusion Matrix)"""
        for method in self._methods:
            print(f"Classifying with Bagging Classifier with base estimator {method}")
            self._bg = BaggingClassifier(base_estimator=method)

            x_train, x_test, y_train, y_test = self.splitTrainTestSet(dataFrame)
            self._bg = self._bg.fit(x_train, y_train)
            y_pred = self._bg.predict(x_test)
            self.printStats(y_test, y_pred, f"Bagging Classifier with base estimator {method} ")

    


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