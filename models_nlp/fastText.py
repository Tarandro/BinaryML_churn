from class_models import Model
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.layers import Layer

from hyperopt import hp


class Fasttext_Attention(Model):

    def __init__(self, objective, seed=15, column_text=None, class_weight=None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'Fasttext_Attention'
        self.is_NN = True

    def hyper_params(self, size_params='small'):
        self.size_params = size_params
        if self.size_params == 'small':
            self.parameters = {'hidden_unit': hp.randint('hidden_unit_1', 120, 130),
                               'learning_rate': hp.choice('learning_rate', [1e-2, 1e-3]),
                               'dropout_rate': hp.uniform('dropout_rate', 0.4, 0.5)}
        else:
            self.parameters = {'hidden_unit': hp.randint('hidden_unit_1', 60, 120),
                               'learning_rate': hp.choice('learning_rate', [1e-2, 1e-3]),
                               'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        return self.parameters

    def init_params(self, size_params='small', method_embedding='../input/fasttext-french-2b-300d/cc.fr.300.vec'):
        if size_params == 'small':
            self.embed_size = 300
            self.max_features = 20000
            self.maxlen = 250
        else:
            self.embed_size = 300
            self.max_features = 100000
            self.maxlen = 350
        self.method_embedding = method_embedding

    def preprocessing_fit_transform(self, x, size_params='small',
                                    method_embedding='../input/fasttext-french-2b-300d/cc.fr.300.vec'):
        self.init_params(size_params, method_embedding)
        # Tokenization by tensorflow with vocab size = max_features
        self.tokenizer = Tokenizer(num_words=self.max_features, lower=True, oov_token="<unk>")
        self.tokenizer.fit_on_texts(list(x.iloc[:, self.column_text]))
        tok = self.tokenizer.texts_to_sequences(x.iloc[:, self.column_text])

        self.word_index = self.tokenizer.word_index
        self.vocab_idx_word = {idx: word for word, idx in self.tokenizer.word_index.items()}

        tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')

        x_preprocessed = {"tok": tok}

        # build_embedding_matrix(self):
        embeddings_index = load_embeddings(self.method_embedding)
        self.embedding_matrix = build_matrix(self.word_index, embeddings_index)

        return x_preprocessed

    def preprocessing_transform(self, x):
        tok = self.tokenizer.texts_to_sequences(x.iloc[:, self.column_text])
        tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')
        x_preprocessed = {"tok": tok}
        return x_preprocessed

    def initialize_params(self, x, y, params):

        try:
            if self.size_params == 'small':
                self.p = {'hidden_unit': params['hidden_unit'],
                          'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate']
                          }
            else:
                self.p = {'hidden_unit': params['hidden_unit'],
                          'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate']
                          }
        except:
            self.p = {'hidden_unit': params['hidden_unit'],
                      'learning_rate': params['learning_rate'],
                      'dropout_rate': params['dropout_rate']
                      }
            self.size_params = 'small'

    def model(self):

        token = tf.keras.layers.Input(shape=(self.maxlen,), name="tok")
        inp = {"tok": token}

        # Embedding + vectorization LSTM
        x = Embedding(len(self.word_index) + 1, self.embed_size, weights=[self.embedding_matrix], trainable=True)(token)
        x = Bidirectional(tf.keras.layers.LSTM(int(self.p['hidden_unit']), return_sequences=True))(x)
        # x = Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
        x = Dropout(self.p['dropout_rate'])(x)
        x = Attention(self.maxlen)(x)

        if 'binary_proba' in self.objective:
            out = Dense(1, 'sigmoid')(x)
        else:
            out = Dense(2, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.p['learning_rate'])
        if 'binary_proba' in self.objective:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model


#################
# Help function : Get fasttext pre-training-weight and attention head
#################

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(embed_dir):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embed_dir, 'rt', encoding='utf-8'))
    return embedding_index


def build_embedding_matrix(word_index, embeddings_index, max_features, lower=True, verbose=True):
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items(), disable=not verbose):
        if lower:
            word = word.lower()
        if i >= max_features: continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embeddings_index[word]
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
    return embedding_matrix

#################
# Build Attention Layer tensorflow :
#################

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
