import pandas as pd
import numpy as np

from functools import reduce
import sys

sys.path.append("../")

# local imports
from utils.functions__utils import consecutive_values
from src.config import logger

class ProcessBureau():
    """
    """

    def __init__(self, bureau_df, bureau_balance_df):
        self.bureau_df = bureau_df
        self.bureau_balance_df = bureau_balance_df
    
    def process(self):
        """
        """
        logger.info("Preprocessing dataframe...")
        # For preprocessing
        self.bureau_balance_df.columns = self.bureau_balance_df.columns.str.lower()
        self.bureau_df.columns = self.bureau_df.columns.str.lower()
        
        self.bureau_balance_df["status_numeric"] = self.bureau_balance_df.status.map({"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"X":None,"C":None})
        self.bureau_balance_df["ever_delinquent"] = np.where(self.bureau_balance_df.status != '0',1,0)
        
        aux_c = self.bureau_balance_df[self.bureau_balance_df.status == "0"].groupby("sk_id_bureau").status.size().to_frame().reset_index().rename(columns = {"status":"times_bucket_0"})
        aux_1 = self.bureau_balance_df[self.bureau_balance_df.status == "1"].groupby("sk_id_bureau").status.size().to_frame().reset_index().rename(columns = {"status":"times_bucket_1"})
        aux_2 = self.bureau_balance_df[self.bureau_balance_df.status == "2"].groupby("sk_id_bureau").status.size().to_frame().reset_index().rename(columns = {"status":"times_bucket_2"})
        aux_3 = self.bureau_balance_df[self.bureau_balance_df.status == "3"].groupby("sk_id_bureau").status.size().to_frame().reset_index().rename(columns = {"status":"times_bucket_3"})
        aux_4 = self.bureau_balance_df[self.bureau_balance_df.status == "4"].groupby("sk_id_bureau").status.size().to_frame().reset_index().rename(columns = {"status":"times_bucket_4"})
        aux_5 = self.bureau_balance_df[self.bureau_balance_df.status == "5"].groupby("sk_id_bureau").status.size().to_frame().reset_index().rename(columns = {"status":"times_bucket_5"})
        aux_6 = self.bureau_balance_df[self.bureau_balance_df.status == "6"].groupby("sk_id_bureau").status.size().to_frame().reset_index().rename(columns = {"status":"times_bucket_6"})
        
        
        aux_ever_delinq = self.bureau_balance_df.groupby("sk_id_bureau").ever_delinquent.max().to_frame().reset_index().rename(columns = {"status":"ever_delinquent"})
        aux_ph = self.bureau_balance_df.groupby("sk_id_bureau").status_numeric.mean().to_frame().reset_index().rename(columns = {"status_numeric":"payment_history"})
        aux_consecutive = self.bureau_balance_df.groupby("sk_id_bureau").status_numeric.apply(lambda x:consecutive_values(x, 0)).to_frame().reset_index().rename(columns = {"status_numeric":"consecutive_no_delinq"})
        
        
        ldf_aux = [self.bureau_df,aux_c, aux_1, aux_2, aux_3, aux_4, aux_5, aux_6, aux_ph, aux_consecutive,aux_ever_delinq]
        
        frame = reduce(lambda x, y: pd.merge(x, y, on = "sk_id_bureau", how = "left"), ldf_aux)
    
        return frame