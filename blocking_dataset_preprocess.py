import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

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


df=df[df['longitude_onset']<=-45]
df=df[df['longitude_onset']>=-170]

df=df[df['latitude_onset']<=75]
df=df[df['latitude_onset']>=15]

pd.set_option('display.max_columns', None)
print(df)
print(df.info())
print(df.describe())

sns.lineplot(x='blocking_year',y='block_intensity',data=df)
plt.savefig('plot0.png')