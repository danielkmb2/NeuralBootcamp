import sys

import json
import matplotlib.pyplot as plt
import os
import pandas as pd

from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data, column=0):
    predicted_data = predicted_data[:, column]
    true_data = true_data[:, column]

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    plt.close()


def plot_historials(historials):

    fig, axes = plt.subplots(nrows=len(historials), ncols=1)
    fig.tight_layout()

    for i, history in enumerate(historials):
        plt.subplot(len(historials), 1, i + 1)
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss in fold %d' % i)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

    plt.show()
    plt.close()


def train_network(configs, dataloader):

    # build model
    model = Model(configs['data']['input_mode'], configs['data']['output_mode'])
    model.build_model(configs['model'])

    # in-memory training
    out_seq_len = configs['data']['input_sequence_length'] if configs['data']['output_mode'] == "many_to_many" else 1
    x_train, y_train = dataloader.get_train_data(
        in_seq_len=configs['data']['input_sequence_length'],
        out_seq_len=out_seq_len
    )

    x_test, y_test = dataloader.get_test_data(
        in_seq_len=configs['data']['input_sequence_length'],
        out_seq_len=out_seq_len
    )

    history = model.train(
        x_train, y_train,
        x_test, y_test,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        shuffle=configs['model']['shuffle_training_data'],
        allow_early_stop=configs['training']['allow_early_stop'],
    )

    return model, history


def main():

    """
    Keras Regression Metrics
        •Mean Squared Error: mean_squared_error, MSE or mse
        •Mean Absolute Error: mean_absolute_error, MAE, mae
        •Mean Absolute Percentage Error: mean_absolute_percentage_error, MAPE, mape
        •Cosine Proximity: cosine_proximity, cosine

    Keras Classification Metrics
        •Binary Accuracy: binary_accuracy, acc
        •Categorical Accuracy: categorical_accuracy, acc
        •Sparse Categorical Accuracy: sparse_categorical_accuracy
        •Top k Categorical Accuracy: top_k_categorical_accuracy (requires you specify a k parameter)
        •Sparse Top k Categorical Accuracy: sparse_top_k_categorical_accuracy (requires you specify a k parameter)
    """

    configs = json.load(open(sys.argv[1], 'r'))
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    # check that the given configuration makes sense
    assert configs['data']['output_mode'] in ["many_to_many", "many_to_one"]
    assert configs['data']['input_mode'] in ["many_to_many", "one_to_many"]
    if configs['data']['input_mode'] == "one_to_many":  # 1-1 dense net
        assert configs['data']['input_sequence_length'] == 0
        assert configs['model']['validation_folds'] >= 1
    else:  # lstm mode
        assert configs['data']['input_sequence_length'] > 0
        assert configs['model']['validation_folds'] == 1

    # load datasets
    filename = os.path.join('data', configs['data']['filename'])
    dataframe = pd.read_csv(filename, sep=',', encoding='utf-8')
    if configs['data']['input_mode'] == "many_to_many":
        dataframe.index.name = 'fecha'

    dataloader = DataLoader()
    dataloader.load_dataset(
        dataframe,
        configs['data']['train_test_split'],
        configs['data']['x_cols'],
        configs['data']['y_cols'],
        configs['data']['cathegorical_cols'],
        model_id=configs['model']['model_id'],
        save_dir=configs['model']['save_dir'],
    )

    # start the k-folded cross-validation
    scores = []
    for i in range(configs['model']['validation_folds']):
        print("Training with fold %d" % i)
        model, history = train_network(configs, dataloader)
        loss = history.history['val_loss'][-1]
        scores.append((loss, model, history))
        if configs['model']['validation_folds'] > 1:
            print("Shuffling data for next fold validation!")
            dataloader.shuffle_data()

    sorted(scores, key=lambda x: x[0])
    model = scores[0][1]  # todo: is the last loss metric the best one to sort
    model.save(save_dir=configs['model']['save_dir'], model_id=configs['model']['model_id'])
    print("Best model has %f loss rate!" % scores[0][0])
    plot_historials([x[2] for x in scores])

    # test the thing!
    out_seq_len = configs['data']['input_sequence_length'] if configs['data']['output_mode'] == "many_to_many" else 1
    x_test, y_test = dataloader.get_test_data(
        in_seq_len=configs['data']['input_sequence_length'],
        out_seq_len=out_seq_len
    )
    unscaled_y_test = dataloader.recompose_results(y_test[:, 0, :], side="y").values

    # predict point by point
    print("Plotting predictions point by point on validation set")
    predictions = model.predict_point_by_point(x_test)
    unscaled_predictions = dataloader.recompose_results(predictions[:, 0, :], side="y").values
    plot_results(unscaled_predictions, unscaled_y_test)

    if model.predictions_are_refeedeable():  # only for lstm mode
        print("Plotting predictions as refeeding window on validation set")
        predictions = model.predict_sequence_full(x_test[1, :, :], len(unscaled_y_test))
        unscaled_predictions = dataloader.recompose_results(predictions[:, 0, :], side="y").values
        plot_results(unscaled_predictions, unscaled_y_test)


if __name__ == '__main__':
    main()
