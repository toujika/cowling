#
# reference:
#   http://memo.saitodev.com/home/python_network_programing/
#
# requirement:
#   python3
#
# how to use:
#   server-machine:$ python cowling/forecaster.py -m main -t buf
#   this(python3) :$ python client.py
#   this(python2) :$ python3 client.py

import argparse
import pickle
import socket
from contextlib import closing

import numpy as np

#from cowling import models
from scripts.process import run_bash

TEST_DATA = '/home/iida/ascetic/cowling/data/test_data.csv'
#TEST_DATA = '/home/pi/ascetic/cowling/data/test_data.csv'
HOST = '133.20.164.17'
PORT = 8010
BUF_SIZE = 16384

def max_extract(pred):
  ary = pred.split('\n')
  current_idx = 0
  max_idx = 0
  max_val = 0
  for a in ary:
    if 'wth' in a:
      current_val = float(a.replace('wth:', ''))
      if max_val < current_val:
        max_val = current_val
        max_idx = current_idx
      current_idx += 1
  return max_idx, max_val

def original_to_db(original_data, arg1, arg2):
  ary = original_data.split('\n')
  db_data = []
  for i in range(len(ary)-2, len(ary)-5, -1):
    line = ary[i]
    ary2 = line.split(',')
    for item_idx in [2, 4, 5]:
      db_data.append(float(ary2[item_idx]))
  db_data.append(float(arg1))
  db_data.append(float(arg2))
  return db_data

def main(host, port, buf_size, test_data, save_db, calc_db):
  out, err = run_bash('python3 log_formatter.py')
  original_data = open(test_data).read()
  
  if calc_db:
    with open('dummy.db', 'rb') as f:
      db = pickle.load(f)
    db_data = original_to_db(original_data, 0, 0)
    max_score = 0
    for i, d in enumerate(db):
      np_db_data = np.array([db_data])
      np_d = np.array([d])
      score = np.dot(np_db_data, np_d.T) / (np.linalg.norm(np_db_data) * np.linalg.norm(np_d))
      if max_score < score:
        max_score = score
        max_idx_db = i
    max_idx = db[max_idx_db][-2]
    max_val = db[max_idx_db][-1]
    result = 'attention: {} hour later, {} %'.format(max_idx + 1, max_val)
    print(result)
    with open('output.tmp', 'w') as f:
      f.write(result)
    return

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  with closing(sock):
    sock.connect((socket.gethostbyname('shinano'), port))
    sended = pickle.dumps(original_data)
    print(len(sended))
    sock.send(sended)
    print('sending success...')
    recv = sock.recv(buf_size)
    recv_dec = recv.decode('utf-8')
    max_idx, max_val = max_extract(recv_dec)
    result = 'attention: {} hour later, {} %'.format(max_idx + 1, max_val)
    print(result)
    with open('output.tmp', 'w') as f:
      f.write(result)

    if save_db:
      try:
        with open('dummy.db', 'rb') as f:
          db = pickle.load(f)
      except:
        db = []
      db_data = original_to_db(original_data, max_idx+1, max_val)
      db.append(db_data)
      with open('dummy.db', 'wb') as f:
        pickle.dump(db, f) 
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', '-hs', type=str, default=HOST)
  parser.add_argument('--port', '-p', type=int, default=PORT)
  parser.add_argument('--buf_size', '-b', type=int, default=BUF_SIZE)
  parser.add_argument('--input_data_file', '-i', type=str, default=TEST_DATA)
  parser.add_argument('--save_db', '-s', type=bool, default=False)
  parser.add_argument('--calc_db', '-c', type=bool, default=False)
  args = parser.parse_args()
  main(host=args.host, port=args.port, buf_size=args.buf_size, test_data=args.input_data_file, save_db=args.save_db, calc_db=args.calc_db)
