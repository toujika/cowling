#
# refer to http://memo.saitodev.com/home/python_network_programing/
#
import argparse
import pickle
import socket
from contextlib import closing

#from cowling import models
from scripts.process import run_bash

TEST_DATA = '/home/iida/ascetic/cowling/data/test_data.csv'
HOST = '127.0.0.1'
PORT = 8010
BUF_SIZE = 16384

def main(host, port, buf_size, test_data):
  out, err = run_bash('python log_formatter.py')
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  with closing(sock):
    sock.connect((host, port))
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
    sock.send(pickle.dumps(original_data))
    recv = sock.recv(buf_size)
    print(recv.decode('utf-8'))
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', '-hs', type=str, default=HOST)
  parser.add_argument('--port', '-p', type=int, default=PORT)
  parser.add_argument('--buf_size', '-b', type=int, default=BUF_SIZE)
  parser.add_argument('--input_data_file', '-i', type=str, default=TEST_DATA)
  args = parser.parse_args()
  main(host=args.host, port=args.port, buf_size=args.buf_size, test_data=args.input_data_file)
