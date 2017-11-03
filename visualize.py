import pickle

import numpy as np
import matplotlib.pyplot as plt

from scripts.monitor import Monitor

X = 'p'
Y = 'h'

def get_value(X, ary):
  t, p, h, pred, teach, i = ary
  if X == 't':
    return t
  elif X == 'p':
    return p
  elif X == 'h':
    return h
  else:
    raise

def get_label(X):
  if X == 't':
    return 'Temperature'
  elif X == 'p':
    return 'Pressure'
  elif X == 'h':
    return 'Humidity'
  else:
    raise

def denormalize(X, nrm, val):
  if X == 't':
    prefix = 'tmp'
  elif X == 'p':
    prefix = 'prs'
  elif X == 'h':
    prefix = 'hmd'
  else:
    raise
  
  mx = nrm[prefix + '_max']
  mn = nrm[prefix + '_min']

  return val * (mx -mn) + mn

with open('t_p_h_pred_teach_i.pickle', 'rb') as f1, \
     open('normalizer.pickle', 'rb') as f2:
  tphpti = pickle.load(f1)
  nrm = pickle.load(f2)

monitor = Monitor(len(tphpti))
plt.figure(figsize=(12,9))
x_max = ''
x_min = ''
y_max = ''
y_min = ''
for n, ary in enumerate(tphpti):
  #if n > 10000: break
  monitor.monitor(n)
  t, p, h, pred, teach, i = ary
  #if not i <= 0: continue
  rb = lambda x, threshold: 'r' if x > threshold else 'b'
  og = lambda x, threshold: (255/255, 150/255, 0/255) if x > threshold else 'g'
  x = denormalize(X, nrm, get_value(X, ary))
  y = denormalize(Y, nrm, get_value(Y, ary))
  plt.scatter(x, y, c=rb(pred, 0.5), marker='+')
  plt.scatter(x, y, c=og(teach, 0.5), marker='x')
  if x_max == '':
    x_max = x
    x_min = x
  else:
    if x_max < x:
      x_max = x
    if x_min > x:
      x_min = x
  if y_max == '':
    y_max = y
    y_min = y
  else:
    if y_max < y:
      y_max = y
    if y_min > y:
      y_min = y


plt.xlabel(get_label(X))
plt.ylabel(get_label(Y))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.savefig('hoge.png')
