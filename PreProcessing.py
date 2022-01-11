import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

class PreProcessing:
    """Contains all methods to preprocess data"""
    def __init__(self):
        self._data = None
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, csv):
        """Sets the dataframe"""
        self._data = pd.read_csv(csv)


    def display(self, option=""):
        """Displays the data """
        if option == "":
            print(self._data)
        elif option == "head":
            print(self._data.head())
        elif option == "tail":
            print(self._data.tail())
    
    def check_nan_percentage(self):
        """Check total precentage of nan values and nans per column """
        nan = self._data.isna().sum()
        print("Missing Data per column")
        print(nan)
        

    def drop_cols(self):
        """drops any column with greater than 50%/5000 rows of missing data """
        nan_cols50 = [i for i in self._data.columns if self._data[i].isnull().sum() > 0.50*len(self._data)] # cols missing > 50% of data
        for col in nan_cols50:
            self._data = self._data.drop(col, axis=1)
    
        return self._data

    def preprocess(self):
        """Calls all methods to preprocess data and displays the data before and after"""
        self._data = self.normalize()
        print("Data before preprocessing")
        print("=" *30)
        self.display()
        self.check_nan_percentage()
        self._data = self.drop_cols()
        print("Data after dropping columns")
        print("=" *30)
        self.check_nan_percentage()
        self._data = self.remove_outliers()
        print("Data after removing outliers")
        print("=" *30)
        self.display()
        self._data = self.impute()
        print("Data after preprocessing")
        print("=" *30)
        self.display()
        self.check_nan_percentage()
        self._data.to_csv('cleanedPatientData.csv')
        return self._data
        
    def encodeForModeling(self, dataFrame):
        """Encodes the data for modeling using a label encoder only used for the logistic regression model returns the encoded dataframe"""
        
        label = LabelEncoder()
        
        for col in dataFrame:
            dataFrame[col] = label.fit_transform(dataFrame[col])

        return dataFrame


    def normalize(self):
        """Normalize data by creating nans and replacing values with a standardized one None is replaced with Norm since they mean the same thing in this context """
        self._data = self._data.replace('',np.nan) #replace spaces/empty with nan
        self._data = self._data.replace(' ',np.nan) #replace spaces/empty with nan
        self._data = self._data.replace('?', np.nan)
        self._data['race'] = self._data['race'].apply(lambda val: np.nan if(val == 'Other' ) else val)
        self._data['A1Cresult'] = self._data['A1Cresult'].apply(lambda val: 'Norm' if(val == 'None' ) else val)# mean the same thing so standardize
        self._data['max_glu_serum'] = self._data['max_glu_serum'].apply(lambda val: 'Norm' if(val == 'None' ) else val)# mean the same thing so standardize

        return self._data
    
    def impute(self):
        """impute/fill missing data using mean for numeric fields and the most frequent value for categorical fields"""
       
        imputerCategorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputerNumeric = SimpleImputer(missing_values=np.nan, strategy='mean')
        categorical = ["race", "gender", "age", "weight", "admission_type_id", "discharge_disposition_id", "admission_source_id", "payer_code", "medical_specialty", "max_glu_serum","A1Cresult", "diabetesMed", "readmitted", "diag_1", "diag_2", "diag_3"]
        for col in self._data:
            if col in categorical:
                self._data[col] = imputerCategorical.fit_transform(self._data[[col]]).ravel()
            else:
                self._data[col] = imputerNumeric.fit_transform(self._data[[col]])

        self._data["time_in_hospital"] = self._data["time_in_hospital"].astype(int)
        self._data["num_lab_procedures"] = self._data["num_lab_procedures"].astype(int)
        self._data["num_procedures"] = self._data["num_procedures"].astype(int)
        self._data["num_medications"] = self._data["num_medications"].astype(int)
        self._data["number_outpatient"] = self._data["number_outpatient"].astype(int)
        self._data["number_emergency"] = self._data["number_emergency"].astype(int)
        self._data["number_inpatient"] = self._data["number_inpatient"].astype(int)
        self._data["number_diagnoses"] = self._data["number_diagnoses"].astype(int)

        return self._data
       
    def get_outliers(self):
        """Returns all outliers using density based spatial clustering 
           min_samples is 750 and eps is 0.4 for outlier detection """
        scaler = MinMaxScaler()
        columns = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient"]

        dataSeries = scaler.fit_transform(self._data[columns])

        outlier_detection = DBSCAN(min_samples=750, eps=0.4)
        clusters = outlier_detection.fit_predict(dataSeries)

        outliers = self._data.iloc[(clusters == -1).nonzero()]
        print("Outliers to be removed")
        print("=" *30)
        print(outliers)
        return outliers

    def remove_outliers(self):
        """Removes the indexes found from get_outliers from the dataframe """
        outliers = self.get_outliers()

        for row in outliers.index:
            self._data = self._data.drop(row)

        return self._data


                
        
        

        
        
        
        
        
       
        
       
        

        
    
    
