import glob

DATA_LOG = '/home/iida/ascetic/cowling/temperature_log'
OUTPUT_FILE = '/home/iida/ascetic/cowling/data/test_data.csv'
HEADER = ',年月日時,気温(℃),降水量(mm),現地気圧(hPa),相対湿度(％)'
NUM_TAIL = 50

def extract_datetime(file_):
  year = file_[4:8]
  month = file_[8:10]
  day = file_[10:12]
  hour = file_[12:14]
  minute = file_[14:16]
  second = file_[16:]
  return '{}/{}/{} {}:{}:{}'.format(year, int(month), int(day), hour, minute, second)

def extract_data(data):
  for i, d in enumerate(data):
    if i == 0:
      tmp = d.split(':')[1].replace('℃', '').replace(' ', '')
    elif i == 1:
      prs = d.split(':')[1].replace('hPa', '').replace(' ', '')
    elif i == 2:
      hmd = d.split(':')[1].replace('％', '').replace(' ', '')
    else:
      pass
  rain = -8888 # dummy number
  return '{},{},{},{}'.format(tmp, rain, prs, hmd)

def tail(num_tail):
  t, e = run_bash('tail -n {} {}'.format(num_tail, OUTPUT_FILE))
  h, e = run_bash('head -n 1 {}'.format(OUTPUT_FILE))
  with open(OUTPUT_FILE, 'w') as f:
    f.write(h + t)

files = sorted(glob.glob('{}/*'.format(DATA_LOG)))

output = '' + HEADER + '\n'
for i, file_ in enumerate(files):
  data = open(file_).read().split('\n')
  idx = i + 1
  datetime = extract_datetime(file_.split('/')[-1])
  extracted_data = extract_data(data)
  formatted_data = '{},{},{}\n'.format(idx, datetime, extracted_data)
  output += formatted_data

with open(OUTPUT_FILE, 'w') as f:
  f.write(output)

tail(NUM_TAIL)
