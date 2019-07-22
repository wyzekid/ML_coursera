from collections import defaultdict
from datetime import datetime, timedelta, date, time

INTERVAL = 10


assert 0 < INTERVAL <= 60

"""get_time_list возвращает массив дата-время, значения которого отличаются на фиксированный интервал времени"""

def get_time_list(start, end, interval):

    t = start + timedelta(seconds=(interval - ((start.second % interval) or interval)))
    res = []

    while t <= end:
        res.append(t.isoformat())
        t = t + timedelta(seconds=interval)

    return res


with open('calls_rnd.txt', 'r') as f:
    res = defaultdict(int)

    for line in f.readlines():
        start_date, start_time, duration = line.split()
        year, month, day = map(int, start_date.split('-'))
        hour, minute, second = map(int, start_time.split(':'))
        duration = int(duration)
        d = date(year, month, day)
        t = time(hour, minute, second)
        start_time = datetime.combine(d, t)
        end_time = start_time + timedelta(seconds=duration)
        #print(start_time, '\t', duration, '\t', end_time)
        time_list = get_time_list(start_time, end_time, INTERVAL)

        for t in time_list:
            res[t] += 1
            #print(t, res[t])

    result = max(res.items(), key=lambda x: x[1])
    print(result)
    print(2**2019%11,2**2021%11)