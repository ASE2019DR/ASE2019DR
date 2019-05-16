from __future__ import absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda
from keras.layers import Dot, add
from keras.models import Input, Model, load_model
from keras import backend as K
import math

os.environ["CUDA_VISIBLE_DEVICES"]="2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


def loss_c(similarity):
    loss_amount = K.log(1 + add([add([add([add([K.exp(-1 * add([similarity[0], -1*similarity[1]])),
                                                K.exp(-1 * add([similarity[0], -1*similarity[2]]))]),
                                           K.exp(-1 * add([similarity[0], -1*similarity[3]]))]),
                                      K.exp(-1 * add([similarity[0], -1*similarity[4]]))]),
                                 K.exp(-1 * add([similarity[0], -1 * similarity[5]]))]))
    return loss_amount


def nagetive_samples_model(input_length, input_dim, output_length, output_dim, hidden_dim):
    q_encoder_input = Input(shape = (input_length, input_dim))
    encoder = GRU(hidden_dim)
    q_encoder_output = encoder(q_encoder_input)
    r_encoder_input = Input(shape = (output_length, output_dim))
    w1_encoder_input = Input(shape = (output_length, output_dim))
    w2_encoder_input = Input(shape=(output_length, output_dim))
    w3_encoder_input = Input(shape=(output_length, output_dim))
    w4_encoder_input = Input(shape=(output_length, output_dim))
    w5_encoder_input = Input(shape=(output_length, output_dim))
    decoder = GRU(hidden_dim)
    r_encoder_output = decoder(r_encoder_input)
    w1_encoder_output = decoder(w1_encoder_input)
    w2_encoder_output = decoder(w2_encoder_input)
    w3_encoder_output = decoder(w3_encoder_input)
    w4_encoder_output = decoder(w4_encoder_input)
    w5_encoder_output = decoder(w5_encoder_input)
    similarity_1 = Dot(axes= 1, normalize=True)([q_encoder_output, r_encoder_output])
    similarity_2 = Dot(axes= 1, normalize=True)([q_encoder_output, w1_encoder_output])
    similarity_3 = Dot(axes= 1, normalize=True)([q_encoder_output, w2_encoder_output])
    similarity_4 = Dot(axes= 1, normalize=True)([q_encoder_output, w3_encoder_output])
    similarity_5 = Dot(axes= 1, normalize=True)([q_encoder_output, w4_encoder_output])
    similarity_6 = Dot(axes=1, normalize=True)([q_encoder_output, w5_encoder_output])
    loss_data = Lambda(lambda x: loss_c(x))([similarity_1, similarity_2, similarity_3, similarity_4, similarity_5, similarity_6])
    model = Model([q_encoder_input, r_encoder_input, w1_encoder_input, w2_encoder_input, w3_encoder_input, w4_encoder_input, w5_encoder_input], similarity_1)
    model.compile(optimizer="adam", loss = lambda y_true, y_pred: loss_data)
    return model


def get_result(input_file_qa, input_data_set, doc_data_set):
    input_file = open(input_file_qa, encoding="UTF-8")
    question_list = input_file.readlines()
    data_return = np.zeros((5, training_size, max_output_len, 300))
    doc_vectors = []
    for i in range(len(doc_data_set)):
        feature_vec = np.zeros(300)
        for j in range(len(doc_data_set[i])):
            feature_vec = np.add(feature_vec, doc_data_set[i][j])
        feature_vec = np.divide(feature_vec, len(doc_data_set[i]))
        doc_vectors.append(feature_vec)
    for i in range(len(input_data_set)):
        feature_vec = np.zeros(300)
        nagetive_simple_get = []
        nagetive_simple_check = []
        for j in range(len(input_data_set[i])):
            feature_vec = np.add(feature_vec, input_data_set[i][j])
        feature_vec = np.divide(feature_vec, len(input_data_set[i]))
        for j in range(len(doc_vectors)):
            similarity_q_d =  np.dot(feature_vec, doc_vectors[j]) / (np.linalg.norm(feature_vec) * (np.linalg.norm(doc_vectors[j])))
            if j != int(question_list[i*2+1].strip()):
                if len(nagetive_simple_get) < 5:
                    nagetive_simple_get.append(similarity_q_d)
                    nagetive_simple_check.append(j)
                else:
                    for l in range(5):
                        if int(question_list[i*2+1].strip()) != 0:
                            if similarity_q_d < nagetive_simple_get[l]:
                                nagetive_simple_get[l] = similarity_q_d
                                nagetive_simple_check[l] = j
                                break
                        else:
                            if similarity_q_d > nagetive_simple_get[l]:
                                nagetive_simple_get[l] = similarity_q_d
                                nagetive_simple_check[l] = j
                                break
        for j in range(5):
            data_return[j][i] = doc_data_set[nagetive_simple_check[j]]
    return data_return



def get_max_len_input(label_file):
    label_f = open(label_file)
    question_max_length = 0
    while 1:
        question = label_f.readline()
        if not question:
            break
        temp = label_f.readline()
        question = question.strip()
        q_words_list = question.split()
        q_length = len(q_words_list) + 1
        if q_length > question_max_length:
            question_max_length = q_length
    label_f.close()
    return question_max_length


def get_max_len_output(doc_file):
    doc_f = open(doc_file)
    doc_max_length = 0
    while 1:
        doc = doc_f.readline()
        if not doc:
            break
        doc = doc.strip()
        d_words_list = doc.split()
        d_length = len(d_words_list) + 1
        if d_length > doc_max_length:
            doc_max_length = d_length
    doc_f.close()
    return doc_max_length


def prepare_data(max_data_len, preparing_data):
    prepared_data = np.zeros((5, training_size, max_data_len, 300))
    print(prepared_data.shape)
    print(preparing_data.shape)
    prepared_data = prepared_data.tolist()
    preparing_data = preparing_data.tolist()
    for m in range(5):
        for i in range(training_size):
            for j in range(max_data_len):
                for k in range(300):
                    prepared_data[m][i][j][k] = preparing_data[i][m][j][k]
    return np.array(prepared_data)


def get_data_input(label_file):
    print("Getting input data...")
    label_f = open(label_file)
    input_set =[]
    while 1:
        question = label_f.readline()
        if not question:
            break
        temp = label_f.readline()
        question = question.strip()
        words_list = question.split()
        input_list_line = []
        for count in range(len(words_list)):
            input_list_line.append(words_list[count])
        input_set.append(input_list_line)
    label_f.close()
    return input_set


def get_data_output(label_file,doc_file):
    print("Getting output data...")
    output_set = []
    doc_f = open(doc_file)
    label_f = open(label_file)
    docs = doc_f.readlines()
    while 1:
        question = label_f.readline()
        if not question:
            break
        doc_num = label_f.readline()
        doc_num = doc_num.strip()
        doc_num = int(doc_num)
        if doc_num != 0:
            doc = docs[doc_num - 1]
            doc = doc.strip()
            words_list = doc.split()
            output_list_line = []
            for count in range(len(words_list)):
                output_list_line.append(words_list[count])
            output_set.append(output_list_line)
        else:
            output_set.append("N/A")
    doc_f.close()
    label_f.close()
    return output_set


def build_data_set(data_set, types_file, vectors_file, data_length):
    types = open(types_file)
    words_list = types.readlines()
    vectors = open(vectors_file)
    vectors_list = vectors.readlines()
    vector_list = []
    for vector in vectors_list:
        vector_info = vector.split()
        vector_list.append(vector_info)
    len_x = len(data_set)
    data_set_new = np.zeros((len_x, data_length, 300))
    for list_num in range(len(data_set)):
        for word_pos in range(len(data_set[list_num])):
            for check_word_pos in range(len(words_list)):
                if words_list[check_word_pos].strip() == data_set[list_num][word_pos].strip():
                    for vector_num_pos in range(len(vector_list[check_word_pos])):
                        data_set_new[list_num, word_pos, vector_num_pos] = vector_list[check_word_pos][vector_num_pos]
    return data_set_new


def get_test_data_input(label_file):
    label_f = open(label_file)
    input_set =[]
    while 1:
        question = label_f.readline()
        if not question:
            break
        temp = label_f.readline()
        question = question.strip()
        words_list = question.split()
        input_list_line = []
        for count in range(len(words_list)):
            input_list_line.append(words_list[count])
        input_set.append(input_list_line)
    label_f.close()
    return input_set


def get_test_data_output(label_file,doc_file):
    output_set = []
    doc_f = open(doc_file)
    label_f = open(label_file)
    docs = doc_f.readlines()
    while 1:
        question = label_f.readline()
        if not question:
            break
        doc_num = label_f.readline()
        doc_num = doc_num.strip()
        doc_num = int(doc_num)
        if doc_num != 0:
            doc = docs[doc_num - 1]
            doc = doc.strip()
            words_list = doc.split()
            output_list_line = []
            for count in range(len(words_list)):
                output_list_line.append(words_list[count])
            output_set.append(output_list_line)
        else:
            output_set.append("N/A")
    doc_f.close()
    label_f.close()
    return output_set


def get_output_data_test():
    doc_data_set = np.load("doc_vec_01.npy")
    result_doc = np.zeros((len(doc_data_set), testing_size))
    result_doc = result_doc.tolist()
    for i in range(len(doc_data_set)):
        for j in range(testing_size):
            result_doc[i][j] = doc_data_set[i]
    return result_doc


batch_size_num = 64
epoch_num = 50
idf_f = "IDF.txt"
hidden_d = 300
testing_size = 201
training_size = 1805
output_file = "doc_list.txt"
input_file = "label_output.txt"
types_input = "types.txt"
vectors_input = "vectors.txt"
max_output_len = get_max_len_output(output_file)
max_input_len = get_max_len_input(input_file)
'''
for temp_num in range(1):
    print("==========Calculating the " + str(temp_num+1) + " data set==========")
    input_data_file = "training-" + str(temp_num+1) + ".txt"
    test_input_data_file = "testing-" + str(temp_num+1) + ".txt"
    input_list = get_data_input(input_data_file)
    test_input_list = get_test_data_input(test_input_data_file)
    output_list = get_data_output(input_data_file, output_file)
    test_output_list = get_test_data_output(test_input_data_file, output_file)
    print("Building training data set...")
    input_data_set = build_data_set(input_list, types_input, vectors_input, max_input_len)
    np.save("input_data_set_" + str(temp_num+1) + ".npy", input_data_set)
    output_data_set = build_data_set(output_list, types_input, vectors_input, max_output_len)
    np.save("output_data_set_" + str(temp_num+1) + ".npy", output_data_set)
    print("Building testing data set...")
    test_input_data_set = build_data_set(test_input_list, types_input, vectors_input, max_input_len)
    np.save("test_input_data_set_" + str(temp_num+1) + ".npy", test_input_data_set)
    test_output_data_set = build_data_set(test_output_list, types_input, vectors_input, max_output_len)
    np.save("test_output_data_set_" + str(temp_num+1) + ".npy", test_output_data_set)
'''
for temp_num in range(1):
    doc_data_set = np.load("doc_vec_01.npy")
    input_data_set = np.load("input_data_set_" + str(temp_num+1) + ".npy")
    output_data_set = np.load("output_data_set_" + str(temp_num+1) + ".npy")
    '''
    question_list_f = "training-" + str(temp_num+1) + ".txt"
    nagetive_simple_list = get_result(question_list_f, input_data_set, doc_data_set)
    nagetive_simple_list = np.array(nagetive_simple_list)
    np.save("nagetive_list_fix_" + str(temp_num+1) + ".npy", nagetive_simple_list)
    model = nagetive_samples_model(max_input_len, hidden_d, max_output_len, hidden_d, hidden_d)
    output_d = []
    for temp in range(training_size):
        temp_set = [1]
        output_d.append(temp_set)
    model.fit([input_data_set, output_data_set, np.array(nagetive_simple_list[0]), np.array(nagetive_simple_list[1]), np.array(nagetive_simple_list[2]), np.array(nagetive_simple_list[3]), np.array(nagetive_simple_list[4])], np.array(output_d), batch_size = batch_size_num, epochs = epoch_num)
    model.save("new_model_" + str(temp_num+1) + ".h5")
    '''
    model = load_model("new_model_" + str(temp_num + 1) + ".h5", custom_objects={'<lambda>': lambda y_true, y_pred: y_true})
    test_input_set = np.load("test_input_data_set_" + str(temp_num+1) + ".npy")
    predicts = {}
    test_data_negative_simples = np.zeros((training_size, max_output_len, 300))
    test_output_set = get_output_data_test()
    for i in range(len(test_output_set)):
        output_data_test = np.array(test_output_set[i])
        predict = model.predict([test_input_set, output_data_test])
        for j in range(len(predict)):
            if j in predicts.keys():
                predicts[j].append(predict[j][0].tolist())
            else:
                predicts[j] = predict[j].tolist()
    output_f = open("new_model_results_fix_" + str(temp_num + 1) + ".txt", "a")
    for i in range(len(predicts)):
        for j in range(len(predicts[i])):
            print(predicts[i][j], file=output_f, end="")
        print("", file=output_f)
    output_f.close()