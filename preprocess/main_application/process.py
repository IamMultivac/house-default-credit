# For preprocessing
import pandas as pd
import numpy as np
# local imports
from src.config import logger

class ProcessMainApplication():
    """
    """
    def __init__(self, frame):
        """
        """
        self.frame = frame

    def _get_age_label(self, x):
        """ Return the age group label (int). """
        age_years = -x / 365
        if age_years < 32.038: 
            return 1
        elif age_years < 39.496: 
            return 2
        elif age_years < 47.178: 
            return 3
        elif age_years < 56.093: 
            return 4
        elif age_years < 69.121: 
            return 5
        else: return 0

    def process(self):
        """
        """
        logger.info("Preprocessing dataframe...")
        self.frame.columns = self.frame.columns.str.lower()
        self.frame["age_range"] = self.frame["days_birth"].apply(lambda x: self._get_age_label(x))
        self.frame['days_employed'].replace(365243, np.nan, inplace=True) 
        self.frame['days_last_phone_change'].replace(0, np.nan, inplace=True) 
        self.frame["code_gender"].replace('XNA',np.nan, inplace=True)

        self.frame = self.frame.reset_index(drop = True)

        return self.frame
