'''
Script for the first EDA of the event data files. The objective of this script
is an attempt to understand the internal structure, value behavior and overall 
usage of the files.
'''
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import seaborn as sbn


def read_head_matches(matches_parquet_pathway: str, output_path: str, engine='pyarrow'):
    matches_df=pd.read_parquet(matches_parquet_pathway, engine=engine)
    matches_df_head=matches_df.head(5).to_json()

    if output_path:
        with open(output_path, 'w') as f:
            f.write(matches_df_head)

    return matches_df_head

def create_matches_df(matches_parquet_pathway: str, engine='pyarrow'):
    matches_df=pd.read_parquet(matches_parquet_pathway, engine=engine)
    return matches_df

def filter_matches_america(matches_df: pd.DataFrame):
    america_matches_df=matches_df[(matches_df['home_team']=='América') | 
                                  (matches_df['away_team']=='América')]
    return america_matches_df

def get_america_matches_id(america_matches_id: pd.DataFrame):
    return list(america_matches_df['match_id'])

if __name__=="__main__":
    matches_pathway = "../data/matches.parquet"
    output_path = "../data/matches_head.json"

    matches_df=create_matches_df(matches_pathway)
    america_matches_df=filter_matches_america(matches_df)
    print(america_matches_df[['match_id','match_date' ,'home_team', 
                              'away_team']].head(5))
    
    america_matches_id=get_america_matches_id(america_matches_df)
    print(america_matches_id[:])
    

