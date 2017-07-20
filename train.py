"""
Train a classifier using keras
to run: python train.py train_file test_file
"""

import h5py
from argparse import ArgumentParser
import numpy as np
from itertools import cycle

import keras
from keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,History
from numpy import genfromtxt
import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    return parser.parse_args()

def generate(input_file,batch_size=100):
    """
    Returns a generator for a keras model
    (either training or evaluation)

    Args:
        input_file (str): path to your chosen input file
        batch_size (int): number of samples per batch
    Returns:
        tuple of numpy arrays with [X,Y,Z,TX,TY,chi2],[signal]        

    """
    #open your csv file
    train_np = np.genfromtxt(input_file, delimiter=',', skip_header=1, autostrip=True, usecols=[2,3,4,5,6,7,8], dtype=float) 
    #count the # of events
    n_events = train_np.shape[0]
    #avoid partial final batch
    limit = int(n_events / batch_size) * batch_size
    #batch it up!
    for start_index in cycle(range(0, limit, batch_size)):
        sl = slice(start_index,start_index + batch_size)
        data = train_np[sl,0:6]
        label = train_np[sl,6]
        #print [data],[label]
        yield (data,label)

def train(train_file):
    """
    Train the model

    Args:
        train_file (str): path to the training csv file
    Returns:
        keras History callback, trained keras model
    """
    print("try to train!")
    #generate(train_file)    
    model = get_model(6)
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True, monitor='loss')
    train_history = model.fit_generator(generate(train_file), steps_per_epoch=500, callbacks=[],epochs=5)

    return train_history, model

def get_model(n_trk_var):
    """
    Make the model

    Args:
        n_trk_var (int): The number of track variables
    Returns:
        model (Keras Model): The model
    """
    # setup inputs
    tracks = layers.Input(shape=(n_trk_var,), name='tracks')

    # add signal output
    signal = layers.Dense(1, activation='softmax', name='signal')(tracks)

    # build and compile the model
    model = Model(inputs=[tracks], outputs=[signal])
    model.compile(optimizer='adam',
                  loss=['mean_squared_error'],
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    args = get_args()
    train(args.train_file)
