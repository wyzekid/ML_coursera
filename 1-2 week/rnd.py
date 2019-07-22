import random
'''year = random.randint(2010, 2017)
month = random.randint(1, 12)
day = random.randint(1, 31)
hours = random.randint(0, 23)
print (x)'''

import radar

''''# Generate random datetime (parsing dates from str values)
radar.random_datetime(start='2000-05-24', stop='2013-05-24T23:59:59')

# Generate random datetime from datetime.datetime values
radar.random_datetime(
    start = datetime.datetime(year=2000, month=5, day=24),
    stop = datetime.datetime(year=2013, month=5, day=24)
)

# Just render some random datetime. If no range is given, start defaults to
# 1970-01-01 and stop defaults to datetime.datetime.now()'''
data = radar.random_datetime(start='2017-06-01T23:59:59', stop='2017-06-07T23:59:59')
duration = random.randint(300, 3600)
#print(duration)
with open("calls_rnd.txt", "w") as file:
    #print(data, duration, '', file=file)
    for i in range(1000):
        data = radar.random_datetime(start='2017-06-01T23:59:59', stop='2017-06-07T23:59:59')
        duration = random.randint(300, 3600)
        print(data, duration, '', file=file)