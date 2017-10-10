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

def main(host, port, buf_size, test_data):
  out, err = run_bash('python3 log_formatter.py')
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  with closing(sock):
    sock.connect((socket.gethostbyname('shinano'), port))
    """
    t = models.TemperaturePredictor(time=3)
    p = models.PressurePredictor(time=3)
    h = models.HumidityPredictor(time=3)
    w = models.WeatherPredictor(time=1)
    tmp_test_X, tmp_test_y = t.make_data(test_data)
    prs_test_X, prs_test_y = p.make_data(test_data)
    hmd_test_X, hmd_test_y = h.make_data(test_data)
    test_X, test_y, y_datetime = w.make_data(test_data, datetime_flag=True)
    original_data = [tmp_test_X, prs_test_X, hmd_test_X, test_X, test_y, y_datetime]
    """
    original_data = open(test_data).read()
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
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', '-hs', type=str, default=HOST)
  parser.add_argument('--port', '-p', type=int, default=PORT)
  parser.add_argument('--buf_size', '-b', type=int, default=BUF_SIZE)
  parser.add_argument('--input_data_file', '-i', type=str, default=TEST_DATA)
  args = parser.parse_args()
  main(host=args.host, port=args.port, buf_size=args.buf_size, test_data=args.input_data_file)
