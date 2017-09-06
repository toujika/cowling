#
# refer to http://memo.saitodev.com/home/python_network_programing/
#
import socket
from contextlib import closing

from scripts.process import run_bash

def main():
  host = '127.0.0.1'
  port = 8010
  bufsize = 4096

  out, err = run_bash('python log_formatter.py')

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  with closing(sock):
    sock.connect((host, port))
    sock.send(b'30')
    #print(len(sock.recv(bufsize)))
    print(sock.recv(bufsize).decode('utf-8'))
  return

if __name__ == '__main__':
  main()
