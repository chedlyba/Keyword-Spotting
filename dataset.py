import os
import json
import librosa
from tqdm import tqdm
import logging

DATASET_PATH = 'dataset'
JSON_PATH = 'data.json'
SAMPLES = 22050

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):

  data = {
      'mappings' : [],
      'labels' : [],
      'MFCCs' : [],
      'files' : [],
  }

  for i, (dirpath, dirnames, filenames) in enumerate((os.walk(dataset_path))):
    
    if dirpath is not dataset_path:
      label = dirpath.split('/')[-1]
      data['mappings'].append(label)
      print(f'\nPreprocessing: {label}')

      for f in (filenames):
        file_path = os.path.join(dirpath, f)
        
        signal, sample_rate = librosa.load(file_path)

        if len(signal)>= SAMPLES:
          signal = signal[:SAMPLES]

        if len(signal) < SAMPLES:
          continue

        MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc,
                                       n_fft=n_fft, hop_length=hop_length
                                       )
          
        data['MFCCs'].append(MFCCs.T.tolist())
        data['labels'].append(i-1)
        data['files'].append(file_path)

  with open(json_path, 'w') as fp:
    json.dump(data, fp, indent=4) 

if __name__=='__main__':
  print('Preprocessing dataset:')
    
  logging.basicConfig(filename='logs/dataset.py.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
  logger=logging.getLogger(__name__)

  try:
      preprocess_dataset(DATASET_PATH, JSON_PATH)      
  except Exception as err:
      logger.exception(err)
      