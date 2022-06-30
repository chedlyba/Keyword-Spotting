from flask import Flask, request, jsonify
import random
from keyword_spotting_service import KeywordSpottingService
import os


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    # get audio file and save it
    audio_file = request.files['file']
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)

    # invoke kss and predict keyword
    kss = KeywordSpottingService()
    pred_keyword = kss.predict(file_name)

    # remove file
    os.remove(file_name)

    # keyword in json format
    data = {
        'keyword': pred_keyword
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=False)
    