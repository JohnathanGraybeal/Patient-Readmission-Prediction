
from PreProcessing import PreProcessing
from Rules import Rules
from LogisitcRegression import LogReg as lr
from SupportVector import SupportVector as svm
from NeuralNetwork import NeuralNetwork as nn
from Bagging import Bagging as bg
from Util import Utility
import sys
import os
import warnings
if __name__ == "__main__":
   if not sys.warnoptions: #ignore all warnings from models 
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
   print('# --- PreProcessing ---')

   raw_data = PreProcessing()
   raw_data.data = "10kPatients.csv"
   cleaned_data = raw_data.preprocess()
   print('# --- End PreProcessing ---')

   key = input("Press c to clear the screen:  ")
   if key == "c" or key == "C":
      Utility.ClearScreen()


   print('# --- Rule Mining ---')


   rules = Rules(cleaned_data)
   association_rules = rules.get_patterns()
   print(association_rules.head())

   readmittedTrueRules = association_rules[association_rules['consequents'] == frozenset({'readmitted_Yes'})]
   print(readmittedTrueRules.head())
   readmittedTrueRules.to_csv("ReadmittedTrueRules.csv")
   readmittedFalseRules = association_rules[association_rules['consequents'] == frozenset({'readmitted_No'})]
   print(readmittedFalseRules.head())
   readmittedFalseRules.to_csv("ReadmittedFalseRules.csv")
   expired = association_rules[association_rules['consequents'] == frozenset({'discharge_disposition_id_Expired'})]
   print(expired.head())
   expired.to_csv("ExpiredRules.csv")
   print('# --- End Rule Mining ---')
   key = input("Press c clear the screen:  ")
   if key == "c" or key == "C":
      Utility.ClearScreen()
   

   encoded_data = raw_data.encodeForModeling(cleaned_data)

  
   print('# --- Prediction ---')
   vector = svm(cleaned_data)
   vector.classify(cleaned_data)
   vector.getLearningCurve(cleaned_data)

   

   logistic = lr(cleaned_data)
   logistic.classify(encoded_data)
   logistic.getLearningCurve(encoded_data)

  

   network = nn(cleaned_data)
   network.classify(cleaned_data)
   network.getLearningCurve(cleaned_data)

   methods = [vector._svc, logistic._logReg, network._mlp]

   bagging = bg(cleaned_data, methods)
   bagging.classify(cleaned_data)
   print('# --- End Prediction ---')
   


