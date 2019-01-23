import sys

import json
import matplotlib.pyplot as plt
import os
import pandas

from core.data_processor import DataLoader
from core.model import Model


# usage: python prediction_only_example.py
#                               sample_configs/config_multiparams.json
#                               saved_models/config_multiparams-09102018-185633.h5

def plot_results(predicted_data, true_data, column=0):
    predicted_data = predicted_data[:, column]
    true_data = true_data[:, column]

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


configs = json.load(open(sys.argv[1], 'r'))
if not os.path.exists(configs['model']['save_dir']):
    os.makedirs(configs['model']['save_dir'])

model_id = configs['model']['model_id']
save_dir = configs['model']['save_dir']

dataloader = DataLoader()
x_scaler_filename = save_dir + "/" + model_id + "-x.scaler"
y_scaler_filename = save_dir + "/" + model_id + "-y.scaler"
dataloader.restore_scalers(x_scaler_filename, y_scaler_filename)

filename = os.path.join('data', configs['data']['filename'])
dataframe = pandas.read_csv(filename, sep=',', encoding='utf-8')
dataframe.index.name = 'fecha'
x_data = dataframe.get(configs['data']['x_cols'], ).values

in_seq_len = configs['data']['input_sequence_length']
x_data = x_data[:, :]  # pick three sequences to make predictions
input_data = dataloader.prepare_input_data(x_data, in_seq_len)
print("Input vector shape: " + str(x_data.shape))

model_filename = sys.argv[2]
model = Model(configs['data']['output_mode'])
model.load_model(filepath=model_filename)

print("Plotting predictions point by point on validation set")
predictions = model.predict_point_by_point(input_data)
print(predictions.shape)
unscaled_predictions = dataloader.recompose_results(predictions[:, 0, :], side="y")
plot_results(unscaled_predictions, x_data[configs['data']['input_sequence_length']:, :])
