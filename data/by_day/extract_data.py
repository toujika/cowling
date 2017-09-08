import argparse
import subprocess

import pandas as pd


def run_bash(command):
  p = subprocess.Popen(command, shell=True, 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  p.wait()
  out = ''.join([so.decode('utf-8') for so in p.stdout.readlines()])
  err = ''.join([se.decode('utf-8') for se in p.stderr.readlines()])
  if not err == '':
    print(err)
    assert 0
  return (out, err)


def rename(filepath):
  # add suffix
  ary = filepath.split('.')
  dst = filepath.replace('.' + ary[-1], '') + '_ex.' + ary[-1]
  # extract file name
  ary = dst.split('/')
  filename = ary[-1]
  return filename


def main(input_data_file, output_file, working_directory, data_format, header):
  # Make working directory
  out, err = run_bash('ls -la ' + working_directory)
  if not out == '':
    print('working directory is found, therefore remove it...')
    out, err = run_bash('rm -r ' + working_directory)
    assert err == ''
  print('make working directory...')
  out, err = run_bash('mkdir ' + working_directory)
  assert err == ''
  

  # 1. Check and convert character codes
  # Data file character codes is UTF-8 only.
  out, err = run_bash('nkf --guess ' + input_data_file)
  assert err == ''
  if not 'UTF-8' in out:
    print('input_data_file is not UTF-8, convert character codes...')
    out, err = run_bash('nkf -w --overwrite ' + input_data_file)
    assert err == ''
  else:
    print('input_data_file is UTF-8...')

  # 2. Remove header
  # Removed header example is below:
  # ダウンロードした時刻：2017/08/02 20:07:26
  # 
  # ,東京,東京,東京,東京,東京,東京,東京,東京,東京,東京,東京,東京,東京
  print('remove header of {}...'.format(input_data_file))
  out, err = run_bash('tail -n +{0} {1} > {2}/removed_header.tmp'
                      .format(header, input_data_file, working_directory))
  assert err == ''
  
  # 3. Read a data
  # Data format is follow Japan Meteorological Agency Website:
  # http://www.data.jma.go.jp/gmd/risk/obsdl/
  print('load {}...'.format(input_data_file))
  df = pd.read_csv(working_directory + '/removed_header.tmp')

  # 4. Extract 
  print('extracting {}...'.format(input_data_file))
  if data_format == 'time':
    df_ex = df.ix[1:,['年月日時', '気温(℃)',
                      '降水量(mm)', '現地気圧(hPa)', '相対湿度(％)']]
  elif data_format == 'day':
    df_ex = df.ix[1:,['年月日', '平均気温(℃)',
                      '降水量の合計(mm)', '平均現地気圧(hPa)', '平均湿度(％)']]
  else:
    print('allowed data_format is "time" or "day".')
    raise

  # 5. Save
  df_ex.to_csv(output_file)
  print('save as {}.'.format(output_file))
    

if __name__ == '__main__':
  # Parse Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_data_files', '-i', type=str, nargs='*')
  parser.add_argument('--output_directory', '-o', type=str, default='.')
  parser.add_argument('--working_directory', '-w', type=str, default='tmp')
  parser.add_argument('--format', '-f', type=str, default='time', help='--format time | --format day')
  parser.add_argument('--header', '-hd', type=int, default=4, help='the amount of line about header will be removed.')
  args = parser.parse_args()

  for input_data_file in args.input_data_files:
    main(input_data_file,
         '{}/{}'.format(args.output_directory, rename(input_data_file)),
         args.working_directory, args.format, args.header)
    print('-'*30)

  #file_ex = [rename(input_data_file) for input_data_file in args.input_data_files]
  #out, err = run_bash('cat ' + ' '.join(file_ex) + ' > extracted_data.csv')
  #assert err == ''

  file_ex = []
  for i, input_data_file in enumerate(args.input_data_files):
    cmd = ('tail -n +{0} {1} > {2}/extracted_{3}.tmp'
           .format((lambda x: 1 if x == 0 else 2)(i),
           rename(input_data_file), args.working_directory, i))
    out, err = run_bash(cmd)
    assert err == ''
    file_ex.append('{}/extracted_{}.tmp'.format(args.working_directory, i))
  out, err = run_bash('cat ' + ' '.join(file_ex) + ' > extracted_data.csv')
  assert err == ''

  print('extracted_data.csv is made.')
  print('extract_data.py is finished.')
