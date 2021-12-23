import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot

df=pd.read_csv("preprocessed_blocking_data.csv") 

year_counts_df = df.groupby('blocking_year')["length_days"].sum()

print(year_counts_df/365*100)