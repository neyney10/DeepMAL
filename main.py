import pandas as pd
from model import DeepMALRawPackets
import numpy as np
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import ast
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


def read_data():
    filename='./merged_output.csv'
    df = pd.read_csv(filename,
                     usecols=['udps.n_bytes_per_packet', 'application_name'],
                     nrows=50)
    df['udps.n_bytes_per_packet'] = df['udps.n_bytes_per_packet'].apply(ast.literal_eval)
    df['label'] = df['application_name'].apply(lambda elem: 1 if elem == 'TLS.DoH_DoT' else 0)
    
    filename_doh = './merged_output_doh.csv'
    df_doh = pd.read_csv(filename_doh,
                     usecols=['udps.n_bytes_per_packet', 'application_name'],
                     nrows=75)
    df_doh['udps.n_bytes_per_packet'] = df_doh['udps.n_bytes_per_packet'].apply(ast.literal_eval)
    df_doh['label'] = df_doh['application_name'].apply(lambda elem: 1 if elem == 'TLS.DoH_DoT' else 0)
    
    return df[['udps.n_bytes_per_packet', 'label']], df_doh[['udps.n_bytes_per_packet', 'label']]

# Read data
print('[Status] Reading data...')
data, data_doh = read_data()
print(data.memory_usage(index=True, deep=True))
print(data_doh.memory_usage(index=True, deep=True))

deep_mal_classifier = DeepMALRawPackets()
print(deep_mal_classifier.model.summary())

# Training
print('[Status] Training model...')
train_data = pd.concat([data.sample(frac=0.8), data_doh.iloc[:int(len(data_doh.index)/1.5)]]).sample(frac=1)
train_data = train_data.explode('udps.n_bytes_per_packet', ignore_index=True)
print(train_data.memory_usage(index=True, deep=True))
x_train = list(train_data['udps.n_bytes_per_packet'])
y_train = list(train_data['label'])
print('Amount of instances', len(train_data.index))
print('Amount of positive instances:', len(train_data[train_data['label']==1].index))

epochs = 5
batch_size = 32
deep_mal_classifier.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, class_weight={0: 0.32, 1: 0.68}, verbose=2) # {0: 0.012, 1: 0.988}

# predict
print('[Status] Predicting...')
test_data  = pd.concat([data.sample(frac=0.4), data_doh.iloc[int(len(data_doh.index)/1.5):]]).sample(frac=1)
test_data = test_data.explode('udps.n_bytes_per_packet', ignore_index=True)
x_test = list(test_data['udps.n_bytes_per_packet'])
y_test = list(test_data['label'])
predictions = deep_mal_classifier.model.predict(x_test)
predictions[predictions>0.5] = 1
predictions[predictions<=0.5] = 0
print(sum(predictions), len(predictions))
print(confusion_matrix(y_test, predictions))