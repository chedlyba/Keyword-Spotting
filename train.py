import json
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import logging
from datetime import datetime
import sys


DATA_PATH = 'data.json'
SAVED_MODEL_PATH = 'model.h5'

lr= 1e-03
EPOCHS=40
BATCH_SIZE=32
NUM_KEYWORDS=10

def load_dataset(data_path):

    with open(data_path, 'r') as fp:
        data = json.load(fp)

    # extract inputs and tagets

    X = np.array(data['MFCCs'])
    y = np.array(data['labels'])

    return X, y


def get_data_split(data_path):
    print('loading dataset...')
    X, y = load_dataset(data_path)
    print('Done.')
        
    # create train/validation/test split
    print('Creating train/validation/test split.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)

    # convert input to 3D
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print('Done.')

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, output_size, lr=1e-03, error='sparse_categorical_crossentropy'):

    # build network
    model = keras.Sequential()

    # layer 1: Conv
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',
        input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(1e-03)))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2 ,2), padding='same'))
    
    # layer 2: Conv
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-03)))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2 ,2), padding='same'))

    # layer 3: Conv
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-03)))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2 ,2), padding='same'))

    # layer 4: flatten
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(keras.layers.Dense(output_size, activation='softmax'))

    # compile the model
    optim = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optim,loss=error, metrics=['accuracy'])

    # print model overview
    model.summary()

    return model

def main():
    # load train/validation/test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_split(DATA_PATH)

    # build the model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, NUM_KEYWORDS, lr)

    # train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_validation, y_validation)
    )

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test error: {test_error}, test accuracy {test_accuracy}')

    # save the model
    model.save(SAVED_MODEL_PATH)

if __name__=='__main__':
    
    # logging and output to files
    logging.basicConfig(filename='logs/train.py.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)

    if sys.argv[0] == '--log':
        sys.stdout = open('logs/train.py.log', 'a')
        print(f'Job started at {datetime.now()}.')
    try:
        main()
    except Exception as err:
        logger.exception(err)

    if sys.argv[0] == '--log':    
        print(f'Job ended at {datetime.now()}.')
        sys.stdout.close()