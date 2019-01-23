import datetime as dt
import numpy as np
import os
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from matplotlib import pyplot
from numpy import newaxis
from tensorflow.test import is_gpu_available

from core.utils import Timer

if is_gpu_available():
    from keras.layers import CuDNNLSTM as LSTM
else:
    from keras.layers import LSTM as LSTM


class NBatchLogger(Callback):
    def __init__(self, total_epochs, total_batches):
        super().__init__()
        self.total_epochs = total_epochs
        self.total_batches = total_batches

        self.currrent_epoch = 0
        self.current_batch = 0

    def update_msg(self):
        progress = int(((self.currrent_epoch + 1) / self.total_epochs) * 50)
        print("Training " +
              "Epoch: %03d/%03d" % (self.currrent_epoch, self.total_epochs) +
              " Batch: %03d/%03d" % (self.current_batch, self.total_batches) +
              "  |" + ("#" * progress) + ("-" * (50 - progress)) + "| %d%%" % (progress * 2), end='\r')

    def on_batch_end(self, batch, logs=None):
        self.current_batch = batch
        self.update_msg()

    def on_epoch_end(self, epoch, logs=None):
        self.currrent_epoch = epoch
        self.update_msg()


class Model:
    """A class for an building and inferencing an lstm model"""

    def __init__(self, input_mode, output_mode):
        self.model = Sequential()

        # configure output mode
        assert output_mode in ["many_to_one", "many_to_many"]
        assert input_mode in ["one_to_many", "many_to_many"]
        self.output_mode = output_mode
        self.intput_mode = input_mode

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):

        timer = Timer()
        timer.start()

        # compile model graph
        for layer in configs['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None

            if layer['type'] == 'dense':
                activation = layer['activation'] if 'activation' in layer else None
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                return_seq = layer['return_seq'] if 'return_seq' in layer else None
                input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
                input_dim = layer['input_dim'] if 'input_dim' in layer else None
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                dropout_rate = layer['rate'] if 'rate' in layer else None
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['loss'], optimizer=configs['optimizer'])

        print('[Model] Model Compiled')

        timer.stop()

    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size,
              allow_early_stop=True, shuffle=False):

        if self.intput_mode == "one_to_many":
            print('[Model] Many to one input configuration found. Refactoring output shapes...')
            x_train = x_train[:, 0, :]
            x_test = x_test[:, 0, :]

        if self.output_mode == "many_to_one":
            print('[Model] Many to one output configuration found. Refactoring output shapes...')
            y_train = y_train[:, 0, :]
            y_test = y_test[:, 0, :]

        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print("[Model] Input shapes:")
        print("[Model]   - X_TRAIN: " + str(x_train.shape))
        print("[Model]   - Y_TRAIN: " + str(y_train.shape))
        print("[Model]   - X_TEST:  " + str(x_test.shape))
        print("[Model]   - Y_TEST:  " + str(y_test.shape))
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        # save_fname = os.path.join(save_dir, model_id + '-%s.h5' % dt.datetime.now().strftime('%d%m%Y-%H%M%S'))
        callbacks = [
            # ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            NBatchLogger(epochs, batch_size)
        ]
        if allow_early_stop:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=30, verbose=1))
            callbacks.append(EarlyStopping(monitor='loss', patience=10, verbose=1))

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            shuffle=shuffle,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=0
        )
        print()  # for the batch logger \r's :V

        print('[Model] Training Completed.')
        timer.stop()

        return history

    def save(self, save_dir, model_id):
        save_fname = os.path.join(save_dir, model_id + '-%s.h5' % dt.datetime.now().strftime('%d%m%Y-%H%M%S'))
        print(self.model.summary())
        print('[Model] Model saved as %s' % save_fname)
        self.model.save(save_fname)

    def predict_point_by_point(self, data):

        if self.intput_mode == "one_to_many":
            print('[Model] Many to one input configuration found. Refactoring output shapes...')
            data = data[:, 0, :]

        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predictions = self.model.predict(data)
        if self.output_mode == "many_to_one":
            predictions = predictions[:, newaxis, :]

        return predictions

    def predict_sequence_full(self, data, epochs):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        # this works well only with single-column predictions or autoencoder-like configurations

        def validate_input(_data):
            _dims_check = len(_data.shape) == 2
            _steps = _data.shape[0]
            _columns = _data.shape[1]
            _out_shape = self.model.layers[-1].output_shape == (
                (None, _steps, _columns) if self.output_mode == "many_to_many" else (None, _columns))
            _in_shape = self.model.layers[0].input_shape == (None, _steps, _columns)

            return _out_shape and _in_shape and _dims_check

        assert validate_input(data)
        assert self.predictions_are_refeedeable()  # validate we can refeed output with input in the network

        curr_frame = data[newaxis, :, :]
        predicted = []
        for i in range(epochs):
            """
            # testing moving window
            from matplotlib import pyplot
            pyplot.plot(curr_frame[0, :, :])
            pyplot.title("prediction window in step %d" % i, y=0.5, loc='right')
            pyplot.show()
            """
            predicted.append(self.predict_point_by_point(curr_frame))

            curr_frame = curr_frame[:, 1:, :]  # remove oldest timestep
            new = predicted[-1][:, -1, :][:, newaxis, :]  # append prediction to the newer timestep
            curr_frame = np.append(curr_frame, new, axis=1)

        predicted = np.array(predicted)[:, -1, :]
        return predicted

    def predictions_are_refeedeable(self):

        if self.intput_mode == "one_to_many":
            return False  # not a sequence thing

        if self.output_mode == "many_to_many":
            _, _, _out_columns = self.model.layers[-1].output_shape
        elif self.output_mode == "many_to_one":
            _, _out_columns = self.model.layers[-1].output_shape
        else:
            return False

        _, _, _in_columns = self.model.layers[0].input_shape

        return _out_columns == _in_columns
