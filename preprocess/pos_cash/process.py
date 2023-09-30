import pandas as pd
import numpy as np
# local imports
from src.config import logger

class ProcessPosCash():
    """
    """

    def __init__(self, pos_cash_balance_df):
        """
        """
    
        self.pos_cash_balance_df = pos_cash_balance_df
    
    def process(self):
        """
        """
        logger.info("Preprocessing dataframe...")
        self.pos_cash_balance_df.columns = self.pos_cash_balance_df.columns.str.lower()
        
        self.pos_cash_balance_df["par_x_at_y"] = self.pos_cash_balance_df["sk_dpd"] / self.pos_cash_balance_df["months_balance"]
        self.pos_cash_balance_df["dpd_total"] =self.pos_cash_balance_df["sk_dpd"] + self.pos_cash_balance_df["sk_dpd_def"]

        return self.pos_cash_balance_df