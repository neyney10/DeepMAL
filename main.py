import pandas as pd
from model import DeepMALRawPackets
import numpy as np
import ast
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

#mat = np.array(ast.literal_eval('[[22, 3, 1, 2, 0, 1, 0, 1, 252, 3, 3],[1, 2, 3, 4, 5, 6, 7, 8, 999, 10, 11]]'))


def read_data(filename='./output.csv'):
    '''
    Todo: return label as well:
    return df[['udps.n_bytes_per_packet', 'label']]
    '''
    df = pd.read_csv(filename, nrows=50)
    df['udps.n_bytes_per_packet'] = df['udps.n_bytes_per_packet'].apply(ast.literal_eval)
    df['label'] = df['application_name'].apply(lambda elem: 1 if elem == 'DNS.DoH_DoT' else 0)
    return df[['udps.n_bytes_per_packet', 'label']]

data = read_data()
deep_mal_classifier = DeepMALRawPackets()
print(deep_mal_classifier.model.summary())

# Training
batch_size = 64
data = data.explode('udps.n_bytes_per_packet', ignore_index=True)
x_train = list(data['udps.n_bytes_per_packet'])
y_train = list(data['label'])

epochs = 5
deep_mal_classifier.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, class_weight={0: 0.012, 1: 0.88}, verbose=2)
print(deep_mal_classifier.model.predict(x_train))