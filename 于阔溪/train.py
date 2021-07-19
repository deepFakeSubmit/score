import model
import tensorflow as tf
import tensorflow.keras as keras
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import matplotlib.pyplot as plt
from predict import predictModel
from feature_extraction import FeatureExtraction
import numpy as np
from Generate import DataGenerator

epochs = 5
train_dir = r'C:\Users\11201\Downloads\zju_deepfake'


def trainModel(model, train_generator, valid_generator, callbacks):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=callbacks,
        # shuffle=True
    )
    return history


# show the changes of loss and accuracy during training
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


def get_files_and_labels(txt_path, mode):
    X = []
    y = []
    with open(os.path.join(train_dir, txt_path)) as f:
        num = 0
        for line in f.readlines():
            num += 1
            if num % 500 == 0:
                print(num)
            fname, tag = line.split()
            tag = np.array([1, 0]) if tag == "spoof" else np.array([0, 1])

            X.append(os.path.join(train_dir, mode, "flac", "%s.flac" % fname))
            y.append(tag)
        y = np.vstack(y)
        # print(X, y.shape, X[0], y[0])
        return X, y


if __name__ == '__main__':
    print('Building model...')

    # Build model
    model = model.camp_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    # set path of saving model
    logdir = os.path.join('graph_def_and_weights')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    output_model_file = os.path.join(logdir, "zjuCamp_weights.h5")

    print('Start training ...')
    # start training

    mode = input('Select mode: 1.Train 2.Predict\nInput number: ')
    if mode == '1':

        files, labels = get_files_and_labels("train.txt", "train")
        # print(files, labels)
        train_generator = DataGenerator(files, labels)
        files, labels = get_files_and_labels("dev.txt", "dev")
        val_generator = DataGenerator(files, labels)

        for i in range(len(train_generator)):
            x, y = train_generator[i]
            # print(x, y)
            print('%s => %s' % (x.shape, y.shape))
        callbacks = [
            keras.callbacks.TensorBoard(logdir),
            keras.callbacks.ModelCheckpoint(output_model_file,
                                            save_best_only=True,
                                            save_weights_only=True),
            keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
        ]
        # model.load_weights(output_model_file)
        history = trainModel(model, train_generator, val_generator, callbacks=callbacks)
        plot_learning_curves(history, 'accuracy', epochs, 0, 1)
        plot_learning_curves(history, 'loss', epochs, 0, 5)
    elif mode == '2':
        # Only run this mode if you have already finished training your model and saved it.
        predictModel(model, output_model_file)
    else:
        print('Please input the correct number.')

    print('Finish! Exit.')
