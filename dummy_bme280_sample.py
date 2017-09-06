# coding: utf-8$

import random

tmp = 25 + random.randint(-100, 100) / 10
prs = 1000 + random.randint(-10000, 10000) / 1000
hmd = 50 + random.randint(0, 50) / 1
#hmd = 50 + random.randint(-50, 50) / 1

dummy = 'temp : {}  ℃\npressure : {} hPa\nhum : {} ％'.format(tmp, prs, hmd)
print(dummy)
