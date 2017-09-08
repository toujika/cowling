# coding: utf-8


import argparse
import math
import pickle

import keras
from keras import backend as K
from keras import layers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers.core import Dense, Dropout, Flatten, Activation  
from keras.layers.recurrent import LSTM
from keras.models import Sequential, optimizers
from keras.regularizers import l2
import numpy as np
import pandas as pd


class Predictor(object):
  
  def __init__(self, time=200):
      
    self.model = None
    self.normalizer = {'tmp_max':0, 'prs_max':0, 'hmd_max':0,
                       'tmp_min':0, 'prs_min':0, 'hmd_min':0}
    self.LOAD_TRAIN_FROM_PICKLE = False
    self.SAVE_TRAIN_TO_PICKLE = True
    self.DIM_TIME = time

  def learning(self, model, X_train, y_train, output='my_model.h5', epoch=100):
    model.fit(X_train, y_train, batch_size=600, nb_epoch=epoch, validation_split=0.05) 
    model.save(output)
    return model

  def load_model(self, model_file):
    return keras.models.load_model(model_file)

  def normalize(self, vector, mode):
    nrm = self.normalizer
    max_ = mode + '_max'
    min_ = mode + '_min'
    vector = (vector - nrm[min_]) / (nrm[max_] - nrm[min_])
    return vector

  def denormalize(self, vector, mode):
    nrm = self.normalizer
    max_ = mode + '_max'
    min_ = mode + '_min'
    vector = vector * (nrm[max_] - nrm[min_]) + nrm[min_]
    return vector

  def create_model(self, time=1):
    hidden_neurons = 50
    dim_time = time
    dim_parameter = 1
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(dim_time, dim_parameter), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    optimizer = optimizers.RMSprop(clipnorm=1.,clipvalue=0.5)
    model.compile(loss="mse", optimizer=optimizer)
    return model


class HumidityPredictor(Predictor):

  def make_data(self, df_file):

    df = pd.read_csv(df_file)

    step = lambda x: 1 if x > 0 else 0
        
    self.normalizer['tmp_max'] = max(df['気温(℃)'])
    self.normalizer['prs_max'] = max(df['現地気圧(hPa)'])
    self.normalizer['hmd_max'] = max(df['相対湿度(％)'])
    self.normalizer['tmp_min'] = min(df['気温(℃)'])
    self.normalizer['prs_min'] = min(df['現地気圧(hPa)'])
    self.normalizer['hmd_min'] = min(df['相対湿度(％)'])
    nrm = self.normalizer

    X = []
    X_slice = []
    Y = []

    data_length = len(df)
    current_iter = 0
    process_line = self.DIM_TIME
    indicate_span = data_length//10 if data_length >= 10 else 1
    for idx in range(data_length - self.DIM_TIME):
      tmp = df.ix[idx, '気温(℃)'] if not np.isnan(df.ix[idx, '気温(℃)']) else -9999
      prs = df.ix[idx, '現地気圧(hPa)'] if not np.isnan(df.ix[idx, '現地気圧(hPa)']) else -9999
      hmd = df.ix[idx, '相対湿度(％)'] if not np.isnan(df.ix[idx, '相対湿度(％)']) else -9999
      rain = df.ix[idx+1, '降水量(mm)'] if not np.isnan(df.ix[idx+1, '降水量(mm)']) else -9999
          
      next_tmp = df.ix[idx+1, '気温(℃)'] if not np.isnan(df.ix[idx+1, '気温(℃)']) else -9999
      next_prs = df.ix[idx+1, '現地気圧(hPa)'] if not np.isnan(df.ix[idx+1, '現地気圧(hPa)']) else -9999
      next_hmd = df.ix[idx+1, '相対湿度(％)'] if not np.isnan(df.ix[idx+1, '相対湿度(％)']) else -9999

      process_line += 1
      if not (tmp == -9999 or prs == -9999 or hmd == -9999 or rain == -9999 
              or next_tmp == -9999 or next_prs == -9999 or next_hmd == -9999):

        X.append(
            np.array([(np.array([hmd])-nrm['hmd_min'])/(nrm['hmd_max']-nrm['hmd_min'])]
                     ).reshape((1)))
        current_iter += 1
        if self.DIM_TIME <= current_iter:
          # DIM_TIME分のデータから、１つの予測
          X_slice.append(X[current_iter-self.DIM_TIME:current_iter])
          Y.append(np.array([(next_hmd-nrm['hmd_min'])/(nrm['hmd_max']-nrm['hmd_min'])]))
          if not current_iter % indicate_span:
            print("{}/{}".format(current_iter, data_length))
      else:
        print('{} is skipped.'.format(process_line))

    with open('normalizer.pickle', mode='wb') as f:
      pickle.dump(self.normalizer, f)
      
    return (np.array(X_slice), np.array(Y))


class PressurePredictor(Predictor):

  def make_data(self, df_file):

    df = pd.read_csv(df_file)

    step = lambda x: 1 if x > 0 else 0
        
    self.normalizer['tmp_max'] = max(df['気温(℃)'])
    self.normalizer['prs_max'] = max(df['現地気圧(hPa)'])
    self.normalizer['hmd_max'] = max(df['相対湿度(％)'])
    self.normalizer['tmp_min'] = min(df['気温(℃)'])
    self.normalizer['prs_min'] = min(df['現地気圧(hPa)'])
    self.normalizer['hmd_min'] = min(df['相対湿度(％)'])
    nrm = self.normalizer

    X = []
    X_slice = []
    Y = []

    data_length = len(df)
    current_iter = 0
    process_line = self.DIM_TIME
    indicate_span = data_length//10 if data_length >= 10 else 1
    for idx in range(data_length - self.DIM_TIME):
      tmp = df.ix[idx, '気温(℃)'] if not np.isnan(df.ix[idx, '気温(℃)']) else -9999
      prs = df.ix[idx, '現地気圧(hPa)'] if not np.isnan(df.ix[idx, '現地気圧(hPa)']) else -9999
      hmd = df.ix[idx, '相対湿度(％)'] if not np.isnan(df.ix[idx, '相対湿度(％)']) else -9999
      rain = df.ix[idx+1, '降水量(mm)'] if not np.isnan(df.ix[idx+1, '降水量(mm)']) else -9999
          
      next_tmp = df.ix[idx+1, '気温(℃)'] if not np.isnan(df.ix[idx+1, '気温(℃)']) else -9999
      next_prs = df.ix[idx+1, '現地気圧(hPa)'] if not np.isnan(df.ix[idx+1, '現地気圧(hPa)']) else -9999
      next_hmd = df.ix[idx+1, '相対湿度(％)'] if not np.isnan(df.ix[idx+1, '相対湿度(％)']) else -9999

      process_line += 1
      if not (tmp == -9999 or prs == -9999 or hmd == -9999 or rain == -9999 
              or next_tmp == -9999 or next_prs == -9999 or next_hmd == -9999):

        X.append(
            np.array([(np.array([prs])-nrm['prs_min'])/(nrm['prs_max']-nrm['prs_min'])]
                     ).reshape((1)))
        current_iter += 1
        if self.DIM_TIME <= current_iter:
          # DIM_TIME分のデータから、１つの予測
          X_slice.append(X[current_iter-self.DIM_TIME:current_iter])
          Y.append(np.array([(next_prs-nrm['prs_min'])/(nrm['prs_max']-nrm['prs_min'])]))
          if not current_iter % indicate_span:
            print("{}/{}".format(current_iter, data_length))
      else:
        print('{} is skipped.'.format(process_line))

    with open('normalizer.pickle', mode='wb') as f:
      pickle.dump(self.normalizer, f)
      
    return (np.array(X_slice), np.array(Y))


class TemperaturePredictor(Predictor):

  def make_data(self, df_file):

    df = pd.read_csv(df_file)

    step = lambda x: 1 if x > 0 else 0
        
    self.normalizer['tmp_max'] = max(df['気温(℃)'])
    self.normalizer['prs_max'] = max(df['現地気圧(hPa)'])
    self.normalizer['hmd_max'] = max(df['相対湿度(％)'])
    self.normalizer['tmp_min'] = min(df['気温(℃)'])
    self.normalizer['prs_min'] = min(df['現地気圧(hPa)'])
    self.normalizer['hmd_min'] = min(df['相対湿度(％)'])
    nrm = self.normalizer

    X = []
    X_slice = []
    Y = []

    data_length = len(df)
    current_iter = 0
    process_line = self.DIM_TIME
    indicate_span = data_length//10 if data_length >= 10 else 1
    for idx in range(data_length - self.DIM_TIME):
      tmp = df.ix[idx, '気温(℃)'] if not np.isnan(df.ix[idx, '気温(℃)']) else -9999
      prs = df.ix[idx, '現地気圧(hPa)'] if not np.isnan(df.ix[idx, '現地気圧(hPa)']) else -9999
      hmd = df.ix[idx, '相対湿度(％)'] if not np.isnan(df.ix[idx, '相対湿度(％)']) else -9999
      rain = df.ix[idx+1, '降水量(mm)'] if not np.isnan(df.ix[idx+1, '降水量(mm)']) else -9999
          
      next_tmp = df.ix[idx+1, '気温(℃)'] if not np.isnan(df.ix[idx+1, '気温(℃)']) else -9999
      next_prs = df.ix[idx+1, '現地気圧(hPa)'] if not np.isnan(df.ix[idx+1, '現地気圧(hPa)']) else -9999
      next_hmd = df.ix[idx+1, '相対湿度(％)'] if not np.isnan(df.ix[idx+1, '相対湿度(％)']) else -9999

      process_line += 1
      if not (tmp == -9999 or prs == -9999 or hmd == -9999 or rain == -9999 
              or next_tmp == -9999 or next_prs == -9999 or next_hmd == -9999):

        X.append(
            np.array([(np.array([tmp])-nrm['tmp_min'])/(nrm['tmp_max']-nrm['tmp_min'])]
                     ).reshape((1)))
        current_iter += 1
        if self.DIM_TIME <= current_iter:
          # DIM_TIME分のデータから、１つの予測
          X_slice.append(X[current_iter-self.DIM_TIME:current_iter])
          Y.append(np.array([(next_tmp-nrm['tmp_min'])/(nrm['tmp_max']-nrm['tmp_min'])]))
          if not current_iter % indicate_span:
            print("{}/{}".format(current_iter, data_length))
      else:
        print('{} is skipped.'.format(process_line))

    with open('normalizer.pickle', mode='wb') as f:
      pickle.dump(self.normalizer, f)
      
    return (np.array(X_slice), np.array(Y))


class WeatherPredictor(Predictor):

  def make_data(self, df_file, datetime_flag=False):

    df = pd.read_csv(df_file)

    step = lambda x: 1 if x > 0 else 0
        
    self.normalizer['tmp_max'] = max(df['気温(℃)'])
    self.normalizer['prs_max'] = max(df['現地気圧(hPa)'])
    self.normalizer['hmd_max'] = max(df['相対湿度(％)'])
    self.normalizer['tmp_min'] = min(df['気温(℃)'])
    self.normalizer['prs_min'] = min(df['現地気圧(hPa)'])
    self.normalizer['hmd_min'] = min(df['相対湿度(％)'])
    nrm = self.normalizer

    X = []
    X_slice = []
    Y = []
    datetime = []

    data_length = len(df)
    current_iter = 0
    process_line = self.DIM_TIME
    indicate_span = data_length//10 if data_length >= 10 else 1
    for idx in range(data_length - self.DIM_TIME):
      tmp = df.ix[idx, '気温(℃)'] if not np.isnan(df.ix[idx, '気温(℃)']) else -9999
      prs = df.ix[idx, '現地気圧(hPa)'] if not np.isnan(df.ix[idx, '現地気圧(hPa)']) else -9999
      hmd = df.ix[idx, '相対湿度(％)'] if not np.isnan(df.ix[idx, '相対湿度(％)']) else -9999
      rain = df.ix[idx+1, '降水量(mm)'] if not np.isnan(df.ix[idx+1, '降水量(mm)']) else -9999
      
      next_tmp = df.ix[idx+1, '気温(℃)'] if not np.isnan(df.ix[idx+1, '気温(℃)']) else -9999
      next_prs = df.ix[idx+1, '現地気圧(hPa)'] if not np.isnan(df.ix[idx+1, '現地気圧(hPa)']) else -9999
      next_hmd = df.ix[idx+1, '相対湿度(％)'] if not np.isnan(df.ix[idx+1, '相対湿度(％)']) else -9999
      next_datetime = df.ix[idx+1, '年月日時'] 

      process_line += 1
      if not (tmp == -9999 or prs == -9999 or hmd == -9999 or rain == -9999 
              or next_tmp == -9999 or next_prs == -9999 or next_hmd == -9999):

        X.append(
            np.array([(np.array([tmp])-nrm['tmp_min'])
                        /(nrm['tmp_max']-nrm['tmp_min']),
                      (np.array([prs])-nrm['prs_min'])
                        /(nrm['prs_max']-nrm['prs_min']),
                      (np.array([hmd])-nrm['hmd_min'])
                        /(nrm['hmd_max']-nrm['hmd_min'])]
            ).reshape((3))
        )
        current_iter += 1
        if self.DIM_TIME <= current_iter:
            # DIM_TIME分のデータから、１つの予測
            X_slice.append(
                X[current_iter-self.DIM_TIME:current_iter]
            )
            Y.append(np.array([step(rain)]))
            datetime.append(next_datetime)
        if not current_iter % indicate_span:
            print("{}/{}".format(current_iter, data_length))
      else:
        print('{} is skipped.'.format(process_line))

    with open('normalizer.pickle', mode='wb') as f:
        pickle.dump(self.normalizer, f)

    if datetime_flag == False:
      return (np.array(X_slice), np.array(Y))
    else:
      return (np.array(X_slice), np.array(Y), datetime)

  def create_model(self, time=1):
    hidden_neurons = 50
    dim_time = time
    dim_parameter = 3
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(dim_time, dim_parameter), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    optimizer = optimizers.RMSprop(clipnorm=1.,clipvalue=0.5)
    model.compile(loss="mse", optimizer=optimizer)
    return model


class WeatherPredictorByDay(Predictor):

  def make_data(self, df_file):

    df = pd.read_csv(df_file)

    step = lambda x: 1 if x > 0 else 0
    
    self.normalizer['tmp_max'] = max(df['平均気温(℃)'])
    self.normalizer['prs_max'] = max(df['平均現地気圧(hPa)'])
    self.normalizer['hmd_max'] = max(df['平均湿度(％)'])
    self.normalizer['tmp_min'] = min(df['平均気温(℃)'])
    self.normalizer['prs_min'] = min(df['平均現地気圧(hPa)'])
    self.normalizer['hmd_min'] = min(df['平均湿度(％)'])
    nrm = self.normalizer

    X = []
    Y = []

    for idx in range(len(df)):
        X.append(np.array(
            [(np.array([df.ix[idx, '平均気温(℃)']])-nrm['tmp_min'])/(nrm['tmp_max']-nrm['tmp_min']),
             (np.array([df.ix[idx, '平均現地気圧(hPa)']])-nrm['prs_min'])/(nrm['prs_max']-nrm['prs_min']),
             (np.array([df.ix[idx, '平均湿度(％)']])-nrm['hmd_min'])/(nrm['hmd_max']-nrm['hmd_min'])]).reshape((1,3)))
        Y.append(np.array([step(df.ix[idx, '降水量の合計(mm)'])]))
    
    with open('normalizer_wth.pickle', mode='wb') as f:
        pickle.dump(self.normalizer, f)
        
    return (np.array(X), np.array(Y))

  def create_model(self, time=1):
    hidden_neurons = 50
    dim_time = time
    dim_parameter = 3
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(dim_time, dim_parameter), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    optimizer = optimizers.RMSprop(clipnorm=1.,clipvalue=0.5)
    model.compile(loss="mse", optimizer=optimizer)
    return model


class SinWavePredictor(Predictor):

  def make_data(self):
    """
    Data format is:
      (batch_id, dim_time, dim_parameter)
      (batch_id, output)
    """
    X = []
    y = []
    augment = 100
    batch = 10000 # batch * dim_time * dim_parameter must be equal augment * step
    dim_time = 1
    dim_parameter = 1
    for a in range(augment):
      step = 100
      for i in range(step):
        val = math.sin((math.pi/(step/2))*i)
        if i == step:
          next_val = math.sin((math.pi/(step/2))*0)
        else:
          next_val = math.sin((math.pi/(step/2))*(i+1))
        X.append(val)
        y.append(next_val)

    X = np.array(X).reshape((batch, dim_time, dim_parameter))
    y = np.array(y).reshape((batch, 1))
    return X, y


def normalize(vector):
  """ Legacy code, for SinWavePredictor
  """
  max_val = max(vector)
  min_val = min(vector)
  vector = (vector - min_val) / max_val * 0.8 + 0.1
  #vector = (vector - min_val) / max_val
  return vector


def denormalize(vector, original_vector):
  """ Legacy code, for SinWavePredictor
  """
  max_val = max(original_vector)
  min_val = min(original_vector)
  vector = (vector - 0.1) / 0.8 * max_val + min_val
  #vector = vector * max_val + min_val
  return vector


def n_gram(tensor, N):
  """ Legacy code, for SinWavePredictor
  """
  head = len(tensor)
  copy1 = tensor.copy()
  copy2 = tensor.copy()
  dst = tensor.copy()
  ring_tensor = np.concatenate((copy1, copy2), axis=0)
  tail = len(ring_tensor)
  for n in range(2, N+1):
    past_tensor = ring_tensor[head-(n-1):tail-(n-1)]
    dst = np.concatenate((past_tensor, dst), axis=1)
  return dst


def recursively_predict(models, X, y, first_model_time=1, use_n_gram=False, test_length=500):
  if use_n_gram == True:
    x = X if first_model_time == 1 else n_gram(X, first_model_time)
  else:
    x = X
  preds = [[] for _ in range(len(models))]
  in_ = [0 for _ in range(len(models))]
  inputs = [[] for _ in range(len(models))]
  for i in range(test_length):
    for t, model in enumerate(models):
      if t == 0:
        in_[t] = x[i]
      else:
        #in_[t] = np.concatenate((in_[t-1], rnn.denormalize(np.array([preds[t-1][-1]]), y)), axis=0)
        in_[t] = np.concatenate((in_[t-1], np.array([preds[t-1][-1]])), axis=0)
      _, _, pred = predict_1D(model, in_[t], 1)
      preds[t].append(pred)
      inputs[t].append(in_[t])
  return preds, inputs
  

def predict_1D(model, input_, iter_):
    """For tmp, prs, hmd"""
    plot_x = []
    plot_y = []
    for i in range(iter_):
      pred = model.predict(np.array([input_]))
      input_ = np.append(input_, pred, axis=0)[1:]
      plot_x.append(i)
      plot_y.append(pred[0][0])
    return input_, plot_x, plot_y


if __name__ == '__main__':
  pass
