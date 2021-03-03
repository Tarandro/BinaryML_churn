import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from IPython.core.display import display, HTML


def attention_weight(x, fixed_weights_attention, biais_attention, step_dim):
    """ redo the calculations made in the attention layer to obtain the weights """
    """ fixed_weights_attention (array) : Fixed weight of the learned attention layer
        biais_attention (array) : bias of the learned attention layer
        step_dim (int) : maxlen """
    """ return : weights (array)"""

    features_dim = fixed_weights_attention.shape[0]

    eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                          K.reshape(fixed_weights_attention, (features_dim, 1))), (-1, step_dim))

    eij += biais_attention

    eij = K.tanh(eij)

    a = K.exp(eij)

    a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

    weights = K.expand_dims(a)
    # weighted_input = x * a
    return weights


def extract_influent_word(bml, type_data, n_influent_word, pr):
    """ extraction of influential words : highlight the words with the highest weight
        for 'Fasttext_Attention' and 'BERT' model
    Args:
        bml : class from BinaryML with 'Fasttext_Attention' and 'BERT' model already trained
        type_data (str) : 'train' or 'test', use documents from train or test dataset
        n_influent_word (int) : number of words to highlight by documents
        pr (dict) : (not use for the moment)
    """

    if 'binary_proba' in bml.objective:
        print(" Extraction doesn't work with objective = binary_proba ")
        return None

    if type_data == 'train':
        X = bml.X_train.copy()
        Y = bml.Y_train.copy()
    else:
        X = bml.X_test.copy()
        Y = bml.Y_test.copy()

    dataset = X.iloc[:, bml.column_text].copy().reset_index(drop=True)
    target = Y[bml.target[0]].copy().reset_index(drop=True)

    # tokenize for FastText
    if 'Fasttext_Attention' in bml.models.keys():
        token_d = bml.models['Fasttext_Attention'].preprocessing_transform(X)['tok']

        fixed_weights_attention = bml.models['Fasttext_Attention'].best_model.layers[-2].get_weights()[0]
        features_dim = fixed_weights_attention.shape[0]
        biais_attention = bml.models['Fasttext_Attention'].best_model.layers[-2].get_weights()[1]

        # Extraction Model (outputs layer outputs from bml.models['Fasttext_Attention'].best_model)
        extract_model_fasttext_attention = tf.keras.Model(inputs=bml.models['Fasttext_Attention'].best_model.input,
                                                          outputs=(
                                                          bml.models['Fasttext_Attention'].best_model.layers[-4].output,
                                                          bml.models['Fasttext_Attention'].best_model.layers[-1].output))

    if 'BERT' in bml.models.keys():
        x_preprocessed = bml.models['BERT'].preprocessing_transform(X)
        ids_d = x_preprocessed[0]
        att_d = x_preprocessed[1]
        tok_d = x_preprocessed[2]
        # cha_d = cha

        all_layer_weights_camembert = bml.models['BERT'].best_model.layers[-1].get_weights()[0]
        # Extraction Model (outputs layer outputs from bml.models['BERT'].best_model)
        extract_model_camembert = tf.keras.Model(inputs=bml.models['BERT'].best_model.input,
                                                 outputs=(bml.models['BERT'].best_model.layers[-4].output,
                                                          bml.models['BERT'].best_model.layers[-1].output))

    html = ''
    for k in range(50):

        if len(dataset[k]) > 500: continue  # trop large à afficher

        list_pred = []
        if 'Fasttext_Attention' in bml.models.keys():
            # USE EXTRACT MODEL
            embedding_output, pred_vec = extract_model_fasttext_attention.predict([token_d[k:k + 1]])
            embedding_output = np.squeeze(embedding_output[0])  # dim (MAX_LEN,256)
            pred = np.argmax(pred_vec)
            list_pred.append(pred)
            weights_attention = attention_weight(embedding_output, fixed_weights_attention, biais_attention,
                                                 bml.models['Fasttext_Attention'].maxlen)
            weights_attention = np.squeeze(weights_attention[0])

        if 'BERT' in bml.models.keys():
            # USE EXTRACT MODEL
            embedding_output_3, pred_vec_3 = extract_model_camembert.predict(
                [ids_d[k:k + 1, :], att_d[k:k + 1, :], tok_d[k:k + 1, :]])
            embedding_output_3 = np.squeeze(embedding_output_3[0])  # dim (MAX_LEN,768)
            pred_3 = np.argmax(pred_vec_3)
            list_pred.append(pred_3)
            layer_weights_3 = all_layer_weights_camembert[:, pred_3]
            final_output_3 = np.dot(embedding_output_3, layer_weights_3)

        if target[k] not in list_pred: continue  # ne pas afficher les mal classés

        # DISPLAY TEXT
        # html = ''
        info = 'Train row %i. Predict %s.   True label is %s' % (k, target[k], target[k])
        html += info + '<br><br>'

        if 'Fasttext_Attention' in bml.models.keys():
            idx = list(token_d[k:k + 1][0]).count(0)
            nb_tok = bml.models['Fasttext_Attention'].maxlen - idx
            if nb_tok < n_influent_word * 2:
                n_influent_word__ = int(nb_tok / 2)
            else:
                n_influent_word__ = n_influent_word

            weights_attention = weights_attention[:-idx]
            v = np.argsort(weights_attention)
            mx = weights_attention[v[-1]];
            x = max(-n_influent_word, -len(v))
            mn = weights_attention[v[x]]

            html += '<b>Attention &emsp;&nbsp;:</b>'
            if pred == target[k]:
                list_ = []
                for j in range(len(weights_attention)):
                    x = (weights_attention[j] - mn) / (mx - mn)
                    list_.append(x)
                g = list(np.argsort(list_))[::-1]
                for j in range(len(weights_attention)):
                    if j in g[:n_influent_word__]:
                        x = 1 - g.index(j) * 0.7 / n_influent_word
                    else:
                        x = 0

                    html += ' '
                    html += "<span style='background:{};font-family:monospace'>".format('rgba(255,255,0,%f)' % x)
                    html += bml.models['Fasttext_Attention'].tokenizer.sequences_to_texts(token_d[k:k + 1, j:j + 1])[0]
                    html += "</span>"
            html += "<br>"

        if 'BERT' in bml.models.keys():
            idx = np.sum(att_d[k,])
            if idx < n_influent_word * 2:
                n_influent_word__ = int(idx / 2)
            else:
                n_influent_word__ = n_influent_word

            ## technique 3
            v = np.argsort(final_output_3[:idx - 1])
            mx = final_output_3[v[-1]];
            x = max(-n_influent_word, -len(v))
            mn = final_output_3[v[x]]

            # DISPLAY TEXT
            html += '<b>Camembert :</b>'
            tokenize = bml.models['BERT'].tokenizer.tokenize(bml.models['BERT'].tokenizer.decode(ids_d[k]))
            list_ = []
            if pred_3 == target[k]:
                for j in range(1, idx):
                    x = (final_output_3[j] - mn) / (mx - mn)
                    list_.append(x)
                g = list(np.argsort(list_))[::-1]
                for j in range(1, idx):
                    if j - 1 in g[:n_influent_word__]:
                        x = 1 - g.index(j - 1) * 0.7 / n_influent_word
                    else:
                        x = 0
                    if tokenize[j][0] == '▁':
                        html += ' '
                    html += "<span style='background:{};font-family:monospace'>".format('rgba(255,255,0,%f)' % x)
                    html += bml.models['BERT'].tokenizer.decode([ids_d[k, j]])
                    html += "</span>"
            html += "<br>"

        html += '<br><br><br>'
    display(HTML(html))
    return html