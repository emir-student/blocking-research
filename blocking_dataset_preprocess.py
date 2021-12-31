import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

df=pd.read_excel("blocknh.xlsx", sheet_name=0, header=0, usecols=list(range(1,13+1))) 
#Gave the columns names
column_names = ["region",
                "length_days",
                "time_begin", "day_begin",
                "time_end", "day_end",
                "block_intensity",
                "longitude_onset", "latitude_onset",
                "block_size",
                "season",
                "month",
                "blocking_year"] 

df.columns=column_names

df = df.dropna(subset=["region"])
# Some rows have empty strings, super annoying to find, got rid of those rows
df = df[df["region"] != ' ']
df = df[df["region"] != '']

for c in column_names:
    df[c] = df[c].astype("float64")
#Casted some of the columns such as 'region' as integers since it was more appropriate
df['region'] = df['region'].astype("int")
df['time_begin']= df['time_begin'].astype("int")
df['day_begin'] = df['day_begin'].astype("int")
df['time_end'] = df['time_end'].astype("int")
df['day_end'] = df['day_end'].astype("int")
df['season'] = df['season'].astype("int")
df['month'] = df['month'].astype("int")
df['blocking_year'] = df['blocking_year'].astype("int")

#Fixed a typo in the Excel sheet where the year was recorded as 2105 instead of 2015. 
df["blocking_year"] = df["blocking_year"].replace(to_replace=2105, value=2015) 

df = df [(df['blocking_year'] != 1981) & (df['month'] !=6) & (df['day_begin'] != 31) ]
df = df [(df['blocking_year'] != 1992) & (df['month'] !=2) & (df['day_begin'] != 30) ] 


#df=df[df['longitude_onset']<=40] 
#df=df[df['longitude_onset']>=-170]

#df=df[df['latitude_onset']<=75]
df=df[df['latitude_onset']>=15]

# July 1970 - June 1971 -> 1970
# Jan (1) - June (6) not correct, add 1 from year

pd.set_option('display.max_rows', 30)

month_fix_index = (df["month"] >= 1) & (df["month"] <= 6)
# df[month_fix_index]["blocking_year"] = df[month_fix_index]["blocking_year"]+1
df.loc[month_fix_index, "blocking_year"] = df[month_fix_index]["blocking_year"]+1

#Filters out years before 1979.
df=df[df['blocking_year']>=1979]

df = df.sort_values(by=['blocking_year', 'month', 'day_begin'])
df['event_id'] = np.asarray(range(len(df.index)))

path='/home/emirs/blocking-research/'
df.to_csv(os.path.join(path,r'preprocessed_blocking_data.csv'),index=False)

pd.set_option('display.max_columns', None)
# print(df)
# print(df.info())
# print(df.describe())

#sns.lineplot(x='blocking_year',y='block_intensity',data=df)
#plt.savefig('plot0.png')

#sns.lineplot(x='blocking_year',y='length_days',data=df)
#plt.savefig('plot1.png')

sns.lineplot(x='latitude_onset',y='block_intensity',data=df)
plt.savefig('plot2.png')

#Make one for duration vs year

