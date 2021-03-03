from class_models import Model
from hyperopt import hp
import tensorflow as tf


class SimpleNeuralNetwork(Model):
    """ Combination of 2 or 3 Networks : Dense + Dropout + BatchNormalization """

    def __init__(self, objective, seed = 15, column_text = None, class_weight = None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'SimpleNeuralNetwork'
        self.is_NN = True

    def hyper_params(self, size_params='small'):
        self.size_params = size_params
        if self.size_params == 'small':
            self.parameters = {'hidden_unit_1': hp.randint('hidden_unit_1', 60, 120),
                               'hidden_unit_2': hp.choice('hd_2', [0, hp.uniform('hidden_unit_2', 60, 120)]),
                               'hidden_unit_3': hp.choice('hd_3', [0, hp.uniform('hidden_unit_3', 60, 120)]),
                               'learning_rate': hp.choice('learning_rate', [1e-2, 1e-3]),
                               'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        else:
            self.parameters = {'hidden_unit_1': hp.randint('hidden_unit_1', 60, 120),
                               'hidden_unit_2': hp.choice('hd_2', [0, hp.uniform('hidden_unit_2', 60, 120)]),
                               'hidden_unit_3': hp.choice('hd_3', [0, hp.uniform('hidden_unit_3', 60, 120)]),
                               'learning_rate': hp.choice('learning_rate', [1e-2, 1e-3]),
                               'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
                               'regularizer': hp.uniform('regularizer', 0.000001, 0.00001)}
        return self.parameters

    def initialize_params(self, x, y, params):
        self.dense_shape = x.shape[1]

        hu = [params['hidden_unit_1']] + [params['hidden_unit_' + str(i)] for i in range(2, 4) if
                                          params['hidden_unit_' + str(i)] != 0]

        try:
            if self.size_params == 'small':
                self.p = {'hidden_units': hu,
                          'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate']
                          }
            else:
                self.p = {'hidden_units': hu,
                          'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate'],
                          'regularizer': params['regularizer']
                          }
        except:
            self.p = {'hidden_units': hu,
                      'learning_rate': params['learning_rate'],
                      'dropout_rate': params['dropout_rate']
                      }
            self.size_params = 'small'

    def model(self):
        # Dense input
        inp = tf.keras.layers.Input(shape=(self.dense_shape,), name="inp")
        x = tf.keras.layers.BatchNormalization()(inp)

        for units in self.p['hidden_units']:
            if self.size_params == 'small':
                x = tf.keras.layers.Dense(int(units), activation='relu')(x)
            else:
                x = tf.keras.layers.Dense(int(units), activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(self.p['regularizer']))(x)
            x = tf.keras.layers.Dropout(self.p['dropout_rate'])(x)
            x = tf.keras.layers.BatchNormalization()(x)

        if 'binary_proba' in self.objective:
            out = tf.keras.layers.Dense(1, 'sigmoid')(x)
        else:
            out = tf.keras.layers.Dense(2, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.p['learning_rate'])
        if 'binary_proba' in self.objective:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model