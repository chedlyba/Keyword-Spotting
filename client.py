import requests

URL = 'http://127.0.0.1:5000/predict'
FILE_PATH_TEST='dataset/up/0a7c2a8d_nohash_0.wav'

if __name__ == '__main__':
    audio_file = open(FILE_PATH_TEST, 'rb')
    values = {
        'file': (FILE_PATH_TEST, audio_file, 'audio/wav')
    }
    response = requests.post(URL, files=values)
    data = response.json()

    print(f'Did you say: { data["keyword"] } ?')