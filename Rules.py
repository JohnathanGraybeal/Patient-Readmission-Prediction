import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class Rules:
    """Association rule mining"""

    def __init__(self, data):
        self._data = data
    def get_patterns(self):
        """Generate patterns found for the dataset returns the association rules that were found saves the rules to AssociationRules.csv in current directory"""
        categorical = ["race", "gender", "age",  "admission_type_id", "discharge_disposition_id", "admission_source_id",  "diabetesMed", "readmitted", "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult"]
        
        df = pd.get_dummies(self._data[categorical]) 
        freq_items = apriori(df, min_support=0.3, use_colnames=True,  verbose=1, max_len=None, low_memory=False)
        

        rules = association_rules(freq_items, metric="confidence", min_threshold=0.4)
        rules.to_csv("AssociationRules.csv")
        

        return rules
    
        

        

        
            

       
