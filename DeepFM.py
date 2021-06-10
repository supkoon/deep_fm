import pandas as pd

from dataloader import dataloader
from wide_layer import wide_part
from deep_layer import deep_part
import tensorflow as tf
from tensorflow import keras
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Deep_FM")
    parser.add_argument('--path', nargs = '?' , default = './dataset/',
                        help= "Input data path")
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='choose dataset')
    parser.add_argument('--embedding_size', type= int, default = 8,
                        help = "wide_part embedding_size")
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout rate.')
    parser.add_argument('--epochs', type = int, default = 10,
                        help = "num epochs")
    parser.add_argument('--batch_size', type = int, default = 32,
                        help = "batch_size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs = '?', default= 'adam',
                        help = "Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument('--layers', nargs='+', default=[400, 400, 400],
                        help='num of layers and nodes of each Dense layer ')
    parser.add_argument('--activation', nargs = '?', default = "relu",
                        help = "choose activation function Dense layer")
    parser.add_argument('--patience', type=int, default=10,
                        help='earlystopping patience')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.(1 or 0)')
    parser.add_argument('--test_size',type=float, default=0.1,
                        help='test_size')
    return parser.parse_args()

class deep_FM(keras.Model):
    def __init__(self, V, num_fields, embbeding_lookup_index, layer_list=[400, 400, 400], dropout_rate=0.5,
                 activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.wide_part = wide_part(V, num_fields, embbeding_lookup_index)
        self.deep_part = deep_part(layer_list, dropout_rate, activation)
        self.output_layer = keras.layers.Dense(1, activation="sigmoid",name = "final_output")

    def call(self, inputs):
        # inputs = (None,108)
        wide_output, embeddings = self.wide_part(inputs)
        deep_output = self.deep_part(embeddings)

        concat = keras.layers.Concatenate(axis=1)([wide_output, deep_output])
        wide_deep_output = self.output_layer(concat)
        return wide_deep_output


if __name__ == "__main__":
    args = parse_args()
    embedding_size = args.embedding_size
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    dropout_rate =  args.dropout_rate
    layers = args.layers
    activation = args.activation
    patience = args.patience
    test_size = args.test_size
    loader = dataloader(args.path + args.dataset)

    num_field = loader.get_num_fields()
    embedding_lookup_index = loader.get_embedding_lookup_index()
    model = deep_FM(embedding_size,num_field,embedding_lookup_index, layer_list=layers, dropout_rate=dropout_rate,
                 activation=activation)

    if learner.lower() == "adagrad":
        model.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate), loss=keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.AUC(), keras.metrics.BinaryAccuracy()])
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), lloss=keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.AUC(), keras.metrics.BinaryAccuracy()])
    elif learner.lower() == "adam":
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.AUC(), keras.metrics.BinaryAccuracy()])
    else:
        model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.AUC(), keras.metrics.BinaryAccuracy()])


    early_stopping_callback = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    model_out_file = 'deep_FM%s.h5' % (datetime.now().strftime('%Y-%m-%d-%h-%m-%s'))
    model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)
    X_train,X_test,y_train,y_test = loader.make_binary_set(test_size= test_size)

    if args.out:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_test, y_test), callbacks=[early_stopping_callback,
                                                                                             model_check_cb]
                            )
    else:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_test, y_test), callbacks=[early_stopping_callback]
                            )
    pd.DataFrame(history.history).plot()