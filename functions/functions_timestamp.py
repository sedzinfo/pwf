# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:17:36 2018

@author: Dimitrios Zacharatos
"""
##########################################################################################
# TIMESTAMP
##########################################################################################
from datetime import date,timedelta,datetime
from datetime import datetime,timedelta
import pandas as pd
import time
import math
import random

today=date.today()
timestamp=time.mktime(today.timetuple())
dt_object=date.fromtimestamp(timestamp)
dt_object.strftime('%Y-%m-%d %H:%M:%S')

today=date.today()
ten_days_later=today+timedelta(days=10)
print(ten_days_later)

def timestamp_info(timestamp):
    dt=datetime.fromtimestamp(timestamp)
    date=dt.strftime('%Y-%m-%d')
    time=dt.strftime('%H:%M:%S')
    day_of_week=dt.strftime('%A')
    week_of_year=dt.strftime('%U')
    month=dt.strftime('%B')
    quarter=(dt.month-1)//3+1
    return date,time,day_of_week,week_of_year,quarter,month

timestamp=datetime.now().timestamp()

date,time,day_of_week,week_of_year,quarter,month=timestamp_info(timestamp)
print(f'Date: {date}')
print(f'Time: {time}')
print(f'Day of the week: {day_of_week}')
print(f'Week of the year: {week_of_year}')
print(f'Quarter: {quarter}')
print(f'Month: {month}')

def timestamp_info_df(timestamp):
    dt=datetime.fromtimestamp(timestamp)
    year=dt.strftime('%Y')
    month_n=dt.strftime('%M')
    day=dt.strftime('%d')
    date=dt.strftime('%Y-%m-%d')
    time=dt.strftime('%H:%M:%S')
    hour=dt.strftime('%H')
    minute=dt.strftime('%M')
    second=dt.strftime('%S')
    day_of_week=dt.strftime('%A')
    week_of_year=dt.strftime('%U')
    month=dt.strftime('%B')
    quarter=(dt.month-1)//3+1
    data={'Date':[date],
          'Time':[time],
          'Year':[year],
          'Week_Year':[week_of_year],
          'Month_No':[month_n],
          'Day_No':[day],
          'Hour':[hour],
          'Minute':[minute],
          'Second':[second],
          'Day':[day_of_week],
          'Quarter':[quarter],
          'Month':[month],
          }
    df=pd.DataFrame(data)
    return df

def timestamp_info_df(timestamp):
    dt=[datetime.fromtimestamp(timestamp) for timestamp in timestamp]
    year=[dt.year for dt in dt]
    month_n=[dt.month for dt in dt]
    day=[dt.day for dt in dt]
    date=[dt.strftime('%Y-%m-%d') for dt in dt]
    time=[dt.strftime('%H:%M:%S') for dt in dt]
    hour=[dt.strftime('%H') for dt in dt]
    minute=[dt.strftime('%M') for dt in dt]
    second=[dt.strftime('%S') for dt in dt]
    day_of_week=[dt.strftime('%A') for dt in dt]
    week_of_year=[dt.strftime('%U') for dt in dt]
    month=[dt.strftime('%B') for dt in dt]
    quarter=[math.ceil(dt.month/3) for dt in dt]
    df=pd.DataFrame(list(zip(year,month_n,day,date,time,hour,minute,second,day_of_week,week_of_year,month,quarter)),columns=["year","month_n","day","date","time","hour","minute","second","day_of_week","week_of_year","month","quarter"])
    return df

timestamp_array=[]
for i in range(1000):
    timestamp_array.append(datetime.now().timestamp())

timestamp_info_df(timestamp)
timestamp_info_df(timestamp_array)

datetime.fromtimestamp(timestamp_array)

timestamps=[1640995200,1641081600,1641168000]
datetimes=[datetime.datetime.fromtimestamp(ts) for ts in timestamps]

num_timestamps=1000
start_date=datetime(1980,1,1)
end_date=datetime(2030,1,1)

timestamps=[]
for _ in range(num_timestamps):
    delta=end_date - start_date
    int_delta=(delta.days*24*60*60)+delta.seconds
    random_second=random.randrange(int_delta)
    timestamp=start_date+timedelta(seconds=random_second)
    timestamps.append(timestamp.timestamp())

timestamp_info_df(timestamps)







