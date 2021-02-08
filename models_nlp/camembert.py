from class_models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

import transformers
from hyperopt import hp

class BERT(Model):

    def __init__(self, objective, seed = 15, column_text = None, class_weight = None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'BERT'
        self.is_NN = True

    def hyper_params(self, size_params = 'small'):
        self.size_params = size_params
        if self.size_params == 'small':
            self.parameters = {'learning_rate': hp.choice('learning_rate', [1e-3, 1e-4]),
                               'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        else:
            self.parameters = {'learning_rate': hp.choice('learning_rate', [1e-3, 1e-4]),
                               'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        return self.parameters

    def init_params(self, size_params = 'small', method_embedding = 'CamemBERT'):
        if size_params == 'small':
            self.MAX_LEN = 120
        else:
            self.MAX_LEN = 350

        if method_embedding == 'RoBERTa':
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
            self.model_bert = transformers.TFRobertaModel.from_pretrained('roberta-base')
        elif method_embedding == 'CamemBERT':
            self.tokenizer = transformers.CamembertTokenizer.from_pretrained('jplu/tf-camembert-base')
            self.model_bert = transformers.TFCamembertModel.from_pretrained("jplu/tf-camembert-base")

    def preprocessing_fit_transform(self, x, size_params = 'small', method_embedding = 'CamemBERT'):
        self.init_params(size_params, method_embedding)

        ct = x.shape[0]
        # INPUTS
        ids = np.ones((ct ,self.MAX_LEN) ,dtype='int32')
        att = np.zeros((ct ,self.MAX_LEN) ,dtype='int32')
        tok = np.zeros((ct ,self.MAX_LEN) ,dtype='int32')

        for k in range(ct):

            text = "  " +" ".join(x.iloc[k ,self.column_text].split())

            enc = self.tokenizer.encode(text, max_length = self.MAX_LEN, truncation=True)

            # CREATE BERT INPUTS
            ids[k ,:len(enc)] = enc
            att[k ,:len(enc)] = 1

        x_preprocessed = {"ids" : ids, "att" : att, "tok" : tok}
        return x_preprocessed

    def preprocessing_transform(self, x):
        ct = x.shape[0]
        # INPUTS
        ids = np.ones((ct ,self.MAX_LEN) ,dtype='int32')
        att = np.zeros((ct ,self.MAX_LEN) ,dtype='int32')
        tok = np.zeros((ct ,self.MAX_LEN) ,dtype='int32')

        for k in range(ct):

            text = "  " +" ".join(x.iloc[k ,self.column_text].split())

            enc = self.tokenizer.encode(text, max_length = self.MAX_LEN, truncation=True)

            # CREATE BERT INPUTS
            ids[k ,:len(enc)] = enc
            att[k ,:len(enc)] = 1

        x_preprocessed = {"ids" : ids, "att" : att, "tok" : tok}
        return x_preprocessed

    def initialize_params(self, x, y, params):

        # hu = [params['hidden_unit_1']] + [params['hidden_unit_'+str(i)] for i in range(2,4) if params['hidden_unit_'+str(i)] != 0]

        try:
            if self.size_params == 'small':
                self.p = {'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate']
                          }
            else:
                self.p = {'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate']
                          }
        except:
            self.p = {'learning_rate': params['learning_rate'],
                      'dropout_rate': params['dropout_rate']
                      }
            self.size_params = 'small'

    def model(self):

        input_ids = tf.keras.layers.Input(shape = (self.MAX_LEN, ), dtype=tf.int32, name="ids")
        attention_mask = tf.keras.layers.Input(shape = (self.MAX_LEN, ), dtype=tf.int32, name="att")
        token = tf.keras.layers.Input(shape = (self.MAX_LEN, ), dtype=tf.int32, name="tok")
        inp = {"ids" : input_ids, "att" : attention_mask, "tok" : token}

        # Embedding + vectorization LSTM
        # Camembert_model = transformers.TFCamembertModel.from_pretrained("jplu/tf-camembert-base")
        x = self.model_bert(input_ids ,attention_mask=attention_mask ,token_type_ids=token)

        x = tf.keras.layers.Dropout(self.p['dropout_rate'])(x[0])
        x = tf.keras.layers.GlobalAveragePooling1D()(x)



        if 'binary_proba' in self.objective:
            out = Dense(1 ,'sigmoid')(x)
        else:
            out = Dense(self.nb_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs = inp, outputs = out)

        optimizer = tf.keras.optimizers.Adam(learning_rate = self.p['learning_rate'])
        if 'binary_proba' in self.objective:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model