import argparse
import pickle
import socket
from contextlib import closing

import numpy as np

import models


TRAIN_DATA = '/home/iida/ascetic/cowling/data/data_2015_by_time.csv'
TEST_DATA = '/home/iida/ascetic/cowling/data/data_2016_by_time.csv'
MODEL_DIR = '/home/iida/ascetic/cowling/model/'


def create_train_model(time, mode, output='model'):
  if mode == 'hmd':
    m = models.HumidityPredictor(time=time)
  elif mode == 'prs':
    m = models.PressurePredictor(time=time)
  elif mode == 'tmp':
    m = models.TemperaturePredictor(time=time)
  else:
    raise
  preX, prey = models.SinWavePredictor.make_data(models)
  X, y = m.make_data(TRAIN_DATA)
  sin = models.SinWavePredictor(time=time)
  pre = sin.create_model(time=time)
  pre = m.learning(pre, models.n_gram(preX, time), models.normalize(prey), output=output+'.h5', epoch=10)
  model = m.learning(pre, X, y, output=output+'.h5', epoch=50)
  print(output + '.h5 is created.')


def recreate_all_models():
  for i in range(3, 10):
    create_train_model(time=i, mode='tmp', output='model_tmp' + str(i))
  for i in range(3, 10):
    create_train_model(time=i, mode='prs', output='model_prs' + str(i))
  for i in range(3, 10):
    create_train_model(time=i, mode='hmd', output='model_hmd' + str(i))
  output = 'model_wth_by_time'
  wt = models.WeatherPredictor(time=3)
  X, y = wt.make_data(TRAIN_DATA)
  model_wth_by_time = wt.create_model(time=3)
  model_wth_by_time = wt.learning(model_wth_by_time, X, y, output=output+'.h5', epoch=500)
  print(output + '.h5 is created.')


def evaluate(test_data=TEST_DATA):
  # 1. loading models

  # 1.1. temperature
  t = models.TemperaturePredictor(time=3)
  tmp_test_x, tmp_test_y = t.make_data(test_data)
  tmp_models = []
  for i in range(3, 10):
    tmp_models.append(models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_tmp'+str(i)+'.h5'))

  # 1.2. pressure
  p = models.PressurePredictor(time=3)
  prs_test_x, prs_test_y = p.make_data(test_data)
  prs_models = []
  for i in range(3, 10):
    prs_models.append(models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_prs'+str(i)+'.h5'))

  # 1.3. humidity
  h = models.HumidityPredictor(time=3)
  hmd_test_x, hmd_test_y = h.make_data(test_data)
  hmd_models = []
  for i in range(3, 10):
    hmd_models.append(models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_hmd'+str(i)+'.h5'))
  
  # 1.4 weather
  w = models.WeatherPredictor(time=1) 
  test_X, test_y = w.make_data(test_data) # 1-gram data
  model_wth_by_time = models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_wth_by_time.h5') # 3-gram model

  # 2. evaluation
  model_time = 3
  model_time_wth= 3
  idx_start = 0
  idx_end = len(hmd_test_x)
  #idx_end = 4000
  mother = 0
  child = 0
  over_range = 0
  for idx in range(idx_start, idx_start + idx_end):
    hp, hi= models.recursively_predict(hmd_models, hmd_test_x[idx:idx+1], None, first_model_time=model_time, test_length=1)
    pp, pi= models.recursively_predict(prs_models, prs_test_x[idx:idx+1], None, first_model_time=model_time, test_length=1)
    tp, ti= models.recursively_predict(tmp_models, tmp_test_x[idx:idx+1], None, first_model_time=model_time, test_length=1)
    for i in range(7):
      if model_time + idx + i < len(test_y):
        mother += 1
        if i == 0:
          # tphのテストデータ二次元 ＋ tphの予測一次元
          input_ = np.array([np.concatenate((tmp_test_x[idx:idx+1][0][1:3], tp[0]), axis=0),
                             np.concatenate((prs_test_x[idx:idx+1][0][1:3], pp[0]), axis=0),
                             np.concatenate((hmd_test_x[idx:idx+1][0][1:3], hp[0]), axis=0)]).reshape((1, 3, 3)).T.reshape((1, 3, 3))
        elif i == 1:
          # tphのテストデータ一次元 ＋ tphの予測ニ次元
          # TODO: そのうち一般化
          input_ = np.array([np.concatenate(([tmp_test_x[idx:idx+1][0][2:3]], tp[0:2]), axis=0),
                             np.concatenate(([prs_test_x[idx:idx+1][0][2:3]], pp[0:2]), axis=0),
                             np.concatenate(([hmd_test_x[idx:idx+1][0][2:3]], hp[0:2]), axis=0)]).reshape((1, 3, 3)).T.reshape((1, 3, 3))
        else:
          input_ = np.array([tp[i-model_time_wth+1:i+1],
                             pp[i-model_time_wth+1:i+1],
                             hp[i-model_time_wth+1:i+1]]).reshape((1, 3, 3)).T.reshape((1, 3, 3))
        pred = model_wth_by_time.predict(input_)[0][0]
        t = test_y[model_time_wth+model_time+idx+i][0]
        if not (lambda x: 1 if x >= 0.1 else 0)(pred) == t:
          child += 1
        if test_y[model_time+idx+i+1][0] == 1:
          print('idx-i:{}-{}\ninput_:{}\nwth:{}\nteach[{}]:{}\n'
                .format(idx, i, input_, pred, model_time+idx+i+1, test_y[model_time+idx+i+1]))
      else:
          over_range += 1
  print('error rate is {}/{}'.format(child, mother))


def main(test_data=TEST_DATA):
  # 1. loading models
  print('load models...')

  # 1.1. temperature
  tmp_models = []
  for i in range(3, 10):
    tmp_models.append(models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_tmp'+str(i)+'.h5'))

  # 1.2. pressure
  prs_models = []
  for i in range(3, 10):
    prs_models.append(models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_prs'+str(i)+'.h5'))

  # 1.3. humidity
  hmd_models = []
  for i in range(3, 10):
    hmd_models.append(models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_hmd'+str(i)+'.h5'))
  
  # 1.4 weather
  model_wth_by_time = models.Predictor.load_model(models.Predictor, MODEL_DIR + 'model_wth_by_time.h5') # 3-gram model
  threshold = lambda x: 1 if x >= 0.1 else 0

  # 2. build the server
  host = '127.0.0.1'
  port = 8010
  backlog = 10
  bufsize = 16384

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  with closing(sock):
    sock.bind((host, port))
    sock.listen(backlog)
    while True:
      print('wait for connection...')
      conn, address = sock.accept()
      with closing(conn):
        msg = conn.recv(bufsize)
        data_length = 30
        
        # old code
        #data_length = int(msg.decode('utf-8'))
        #t = models.TemperaturePredictor(time=3)
        #tmp_test_x, tmp_test_y = t.make_data(test_data)
        #tmp_test_x = tmp_test_x[-data_length:]
        #p = models.PressurePredictor(time=3)
        #prs_test_x, prs_test_y = p.make_data(test_data)
        #prs_test_x = prs_test_x[-data_length:]
        #h = models.HumidityPredictor(time=3)
        #hmd_test_x, hmd_test_y = h.make_data(test_data)
        #hmd_test_x = hmd_test_x[-data_length:]
        #w = models.WeatherPredictor(time=1) 
        #test_X, test_y, y_datetime = w.make_data(test_data, datetime_flag=True) # 1-gram data
        #test_X = test_X[-data_length:]
        #test_y = test_y[-data_length:]
        #y_datetime = y_datetime[-data_length:]

        """
        # new code
        data = pickle.loads(msg)
        tmp_test_x = data[0]
        prs_test_x = data[1]
        hmd_test_x = data[2]
        test_x = data[3]
        test_y = data[4]
        y_datetime = data[5]
        """

        # new code ver2
        data = pickle.loads(msg)
        data_length = 50
        with open(test_data, 'w') as f:
          f.write(data)
        t = models.TemperaturePredictor(time=3)
        tmp_test_x, tmp_test_y = t.make_data(test_data)
        tmp_test_x = tmp_test_x[-data_length:]
        p = models.PressurePredictor(time=3)
        prs_test_x, prs_test_y = p.make_data(test_data)
        prs_test_x = prs_test_x[-data_length:]
        h = models.HumidityPredictor(time=3)
        hmd_test_x, hmd_test_y = h.make_data(test_data)
        hmd_test_x = hmd_test_x[-data_length:]
        w = models.WeatherPredictor(time=1) 
        test_X, test_y, y_datetime = w.make_data(test_data, datetime_flag=True) # 1-gram data
        test_X = test_X[-data_length:]
        test_y = test_y[-data_length:]
        y_datetime = y_datetime[-data_length:]

        

        # 2. evaluation
        model_time = 3
        model_time_wth= 3
        idx_start = 0
        idx_end = len(hmd_test_x)
        msg = ''
        msg_full = ''
        for idx in range(idx_start, idx_start + idx_end):
          hp, hi= models.recursively_predict(hmd_models, hmd_test_x[idx:idx + 1], None, first_model_time=model_time, test_length=1)
          pp, pi= models.recursively_predict(prs_models, prs_test_x[idx:idx + 1], None, first_model_time=model_time, test_length=1)
          tp, ti= models.recursively_predict(tmp_models, tmp_test_x[idx:idx + 1], None, first_model_time=model_time, test_length=1)
          for i in range(7):
            if i == 0:
              # tphのテストデータ二次元 ＋ tphの予測一次元
              input_ = np.array([np.concatenate((tmp_test_x[idx:idx + 1][0][1:3], tp[0]), axis=0),
                                 np.concatenate((prs_test_x[idx:idx + 1][0][1:3], pp[0]), axis=0),
                                 np.concatenate((hmd_test_x[idx:idx + 1][0][1:3], hp[0]), axis=0)]).reshape((1, 3, 3)).T.reshape((1, 3, 3))
            elif i == 1:
              # tphのテストデータ一次元 ＋ tphの予測ニ次元
              # TODO: そのうち一般化
              input_ = np.array([np.concatenate(([tmp_test_x[idx:idx + 1][0][2:3]], tp[0:2]), axis=0),
                                 np.concatenate(([prs_test_x[idx:idx + 1][0][2:3]], pp[0:2]), axis=0),
                                 np.concatenate(([hmd_test_x[idx:idx + 1][0][2:3]], hp[0:2]), axis=0)]).reshape((1, 3, 3)).T.reshape((1, 3, 3))
            else:
              input_ = np.array([tp[i-model_time_wth + 1:i + 1],
                                 pp[i-model_time_wth + 1:i + 1],
                                 hp[i-model_time_wth + 1:i + 1]]).reshape((1, 3, 3)).T.reshape((1, 3, 3))
            pred = model_wth_by_time.predict(input_)[0][0]
            # idxを起点に、tphがmodel_time分先の予測を行う。これをi回分行う。
            # さらにwthがmodel_time_wth分先の予測を行う。
            dt = y_datetime[model_time_wth + model_time + idx + i] \
                 if model_time_wth + model_time + idx + i < len(test_y) else y_datetime[-1]
            # 最後のidxの情報が、最新のデータによる６時間先までの予測
            msg += '{}\nidx-i:{}-{}\nwth:{}\n\n'.format(dt, idx, i, threshold(pred)) if idx == idx_start + idx_end - 1 else ''
            #msg_full += '{}\nidx-i:{}-{}\nwth:{}\n\n'.format(dt, idx, i, pred)

        conn.send(msg.encode('utf-8'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_data_file', '-t', type=str)
  parser.add_argument('--mode', '-m', type=str, default='evaluate',
                      help='evaluate | main')
  args = parser.parse_args()

  if args.mode == 'evaluate':
    evaluate()
  else:
    assert args.test_data_file is not None
    main(args.test_data_file)
