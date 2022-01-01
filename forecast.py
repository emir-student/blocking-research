import numpy as np
import datetime

dates = []
start_date = datetime.date(1979, 1, 1) 
end_date = datetime.date(2020, 12, 31)    # perhaps date.now()
delta = end_date - start_date   # returns timedelta
for i in range(delta.days + 1):
    day = start_date + datetime.timedelta(days=i)
    day=day.strftime("%Y-%-m-%-d") #Hopefully will not be necessary in the future once the memory issue is fixed
    dates.append(day)


#reanalysis_fp= []
#for date in dates:
#    array = np.load(f"./reanalysis_data_forecasting_processed/{date}.npy")
#    reanalysis_fp.append(array)

#reanalysis_fp = np.array(reanalysis_fp)

dates = []
start_date = datetime.date(1979, 1, 1) 
end_date = datetime.date(2020, 12, 31)    # perhaps date.now()
delta = end_date - start_date   # returns timedelta
for i in range(delta.days + 1):
    day = start_date + datetime.timedelta(days=i)
    dates.append(day)






