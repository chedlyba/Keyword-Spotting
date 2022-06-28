from tensorflow import keras
import numpy as np
import librosa

MODEL_PATH = 'model.h5'

class _KeywordSpottingService:

    model = None
    _mappings = [
        'right',
        'left',
        'yes',
        'stop',
        'no',
        'up',
        'on',
        'down',
        'off',
        'go'
    ]
    _instance = None

    MAX_SAMPLES = 22050
    n_fft = 2048
    num_mfcc = 13
    hop_length = 512

    def predict(self, file_path):

        # Extract features
        signal, sr = librosa.load(file_path)
        if len(signal) >= self.MAX_SAMPLES:
            signal = signal[:self.MAX_SAMPLES]
        else :
            return None
        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.num_mfcc, # (#segments, #coefficients)
                                       n_fft=self.n_fft, hop_length=self.hop_length
                                       ).T
        
        # Convert 2d MFCCs array into 4d array --> (#samples, #segments, #coefficients, #channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # predict output
        predictions = self.model.predict(MFCCs) 

        index = np.argmax(predictions)
        keyword = self._mappings[index]

        return keyword

def KeywordSpottingService():

    # Ensure the instance is unique
    if _KeywordSpottingService._instance is None:
        _KeywordSpottingService._instance = _KeywordSpottingService()
        _KeywordSpottingService.model = keras.models.load_model(MODEL_PATH)

    return _KeywordSpottingService._instance

if __name__ == '__main__':

    kss = KeywordSpottingService()

    pred1 = kss.predict('dataset/down/0a7c2a8d_nohash_0.wav')
    pred2 = kss.predict('dataset/down/fe291fa9_nohash_0.wav')
    pred3 = kss.predict('dataset/right/0a7c2a8d_nohash_0.wav')
    print(pred1, pred2, pred3)