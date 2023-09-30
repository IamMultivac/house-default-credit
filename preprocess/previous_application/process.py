# For preprocessing
import pandas as pd
import numpy as np

# local imports
from src.config import logger

class ProcessPreviousApplication():
    """
    """
    def __init__(self, frame):
        """
        """
        self.frame = frame


    def process(self):
        """
        """
        logger.info("Preprocessing dataframe...")
        self.frame.columns = self.frame.columns.str.lower()

        self.frame['days_first_drawing'].replace(365243, np.nan, inplace=True)
        self.frame['days_first_due'].replace(365243, np.nan, inplace=True)
        self.frame['days_last_due_1st_version'].replace(365243, np.nan, inplace=True)
        self.frame['days_last_due'].replace(365243, np.nan, inplace=True)
        self.frame['days_termination'].replace(365243, np.nan, inplace=True)

        self.frame["requested_vs_final_amt"] = self.frame["amt_credit"] - self.frame["amt_application"]


        return self.frame
