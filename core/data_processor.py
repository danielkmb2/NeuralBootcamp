import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class DataframeBinarizer:
    def __init__(self):
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.one_hot_cols = []
        self.y_cols = []

    def fit_transform(self, _df, one_hot_cols):
        self.one_hot_cols = list(set(one_hot_cols).intersection(set(_df.columns)))

        for col in self.one_hot_cols:
            self.label_encoders[col] = LabelEncoder()
            self.onehot_encoders[col] = OneHotEncoder()

            _df[col] = self.label_encoders[col].fit_transform(_df[col])
            df_one_hot = self.onehot_encoders[col].fit_transform(_df[col].values.reshape(-1, 1)).toarray()

            onehot_cols = ["_" + col + "_" + str(int(i)) for i in range(df_one_hot.shape[1])]
            df_one_hot = pd.DataFrame(df_one_hot, columns=onehot_cols)
            _df = pd.concat([_df, df_one_hot], axis=1)
            _df.drop([col], axis=1, inplace=True)

        self.y_cols = _df.columns
        return _df

    def transform(self, _df):
        for col in self.one_hot_cols:
            _df[col] = self.label_encoders[col].transform(_df[col])
            df_one_hot = self.onehot_encoders[col].transform(_df[col].values.reshape(-1, 1)).toarray()

            onehot_cols = ["_" + col + "_" + str(int(i)) for i in range(df_one_hot.shape[1])]
            df_one_hot = pd.DataFrame(df_one_hot, columns=onehot_cols)
            _df = pd.concat([_df, df_one_hot], axis=1)
            _df.drop([col], axis=1, inplace=True)

        return _df

    def inverse_transform(self, _df):
        _df = pd.DataFrame(data=_df, columns=self.y_cols)

        for col in self.one_hot_cols:
            onehot_cols = _df.columns[pd.Series(_df.columns).str.startswith("_" + col + "_")]

            df_one_hot = _df[onehot_cols]
            df_one_hot = self.onehot_encoders[col].inverse_transform(df_one_hot)
            df_one_hot = self.label_encoders[col].inverse_transform(df_one_hot.round().astype(int))

            _df.drop(onehot_cols, axis=1, inplace=True)
            _df[col] = df_one_hot

        return _df


class DataLoader:
    """A class for loading and transforming data for the lstm model"""

    def __init__(self):
        self.x_scaler = None
        self.y_scaler = None
        self.x_binarizer = None
        self.y_binarizer = None

        self.split = 0.5
        self.x_data = None
        self.y_data = None

    def load_dataset(self, dataframe, split, x_cols, y_cols, cathegorical_cols, model_id, save_dir):
        self.split = split

        # build x data

        x_df = dataframe.get(x_cols)
        self.x_binarizer = DataframeBinarizer()
        x_df = self.x_binarizer.fit_transform(x_df, cathegorical_cols)

        x_data = x_df.values
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.x_data = self.x_scaler.fit_transform(x_data)
        print(self.x_data.shape)

        # build y data
        y_df = dataframe.get(y_cols)
        self.y_binarizer = DataframeBinarizer()
        y_df = self.y_binarizer.fit_transform(y_df, cathegorical_cols)

        y_data = y_df.values
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_data = self.y_scaler.fit_transform(y_data)

        if save_dir is not None:
            self._save_output_scalers(self.y_scaler, save_dir, model_id)

    def restore_scalers(self, x_scaler_filename, y_scaler_filename):
        self.x_scaler = joblib.load(x_scaler_filename)
        self.y_scaler = joblib.load(y_scaler_filename)
        print("[DataLoader] Scalers loaded!")

    @staticmethod
    def _save_output_scalers(scaler, save_dir, model_id):
        joblib.dump(scaler, save_dir + "/" + model_id + "-x.scaler")
        joblib.dump(scaler, save_dir + "/" + model_id + "-y.scaler")
        print("[DataLoader] Scalers saved!")

    def recompose_results(self, data, *, side):
        assert (side in ["x", "y"])

        scaler = self.x_scaler if side == "x" else self.y_scaler
        inv_yhat = scaler.inverse_transform(data)

        # todo: what about missing and new labels?
        inv_yhat = self.y_binarizer.inverse_transform(inv_yhat)

        return inv_yhat

    def get_test_data(self, in_seq_len, out_seq_len):
        """
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        """
        i_split = int(len(self.x_data) * self.split)
        x_data_test = self.x_data[i_split:]
        y_data_test = self.y_data[i_split:]

        return self._get_data_splits(in_seq_len, out_seq_len, x_data_test, y_data_test)

    def get_train_data(self, in_seq_len, out_seq_len):
        """
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        """
        i_split = int(len(self.x_data) * self.split)
        x_data_train = self.x_data[:i_split]
        y_data_train = self.y_data[:i_split]

        return self._get_data_splits(in_seq_len, out_seq_len, x_data_train, y_data_train)

    def shuffle_data(self):
        stack = np.hstack([self.x_data, self.y_data])
        np.random.shuffle(stack)

        self.x_data = stack[:, :self.x_data.shape[1]]
        self.y_data = stack[:, -self.y_data.shape[1]:]

    @staticmethod
    def _get_sequence_data(data, in_seq_len, out_seq_len):
        data_windows = []
        for i in range(len(data) - (in_seq_len + out_seq_len)):
            data_windows.append(data[i:i + (in_seq_len + out_seq_len)])

        return np.array(data_windows).astype(float)

    @staticmethod
    def _get_data_splits(in_seq_len, out_seq_len, x, y):

        if in_seq_len == 0:
            # dont generate sequences, just format the arrays for consistency
            return x[:, np.newaxis, :], y[:, np.newaxis, :]
        else:
            x = DataLoader._get_sequence_data(x, in_seq_len, out_seq_len)[:, :-out_seq_len, :]
            y = DataLoader._get_sequence_data(y, in_seq_len, out_seq_len)[:, -out_seq_len:, :]
            return x, y

    def prepare_input_data(self, x_data, in_seq_len):
        x_data = self.y_scaler.fit_transform(x_data)

        data_windows = []
        for i in range(len(x_data) - in_seq_len):
            data_windows.append(x_data[i:i + in_seq_len])

        return np.array(data_windows).astype(float)
