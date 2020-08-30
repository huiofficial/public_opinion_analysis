import numpy as np
import re
import os
import pickle
import itertools
from collections import Counter

cop = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9]")
company_x ="../data/formated_data/company/company_x_v1.pkl"
text_y = "../data/formated_data/company/company_texty_v1.pkl"
sen3_y = "../data/formated_data/company/company_sen3_v1.pkl"
sen5_y = "../data/formated_data/company/company_sen5_v1.pkl"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = cop.sub("", string)
    return string.strip().lower()

def load_data_and_labels(dataset_name):
    """
    formated data: x.pkl & y.pkl
    """
    formated_xfile = "../data/formated_data/"+dataset_name+"/x.pkl"
    formated_yfile = "../data/formated_data/"+dataset_name+"/y.pkl"
    
    if os.path.exists(formated_xfile) and os.path.exists(formated_yfile):
        with open(formated_xfile, "rb") as file:
            x = pickle.load(file)
        with open(formated_yfile, "rb") as file:
            y = pickle.load(file)
    else:
        if dataset_name == "mr":
            if os.path.isdir("../data/formated_data/"+dataset_name+"/") == False:
                os.mkdir("../data/formated_data/"+dataset_name+"/")
            x, y = preprocess_mr("../data/rt-polaritydata/rt-polarity.pos.utf8","../data/rt-polaritydata/rt-polarity.neg.utf8")
            with open(formated_xfile, "wb") as file:
                pickle.dump(x, file)
            with open(formated_yfile, "wb") as file:
                pickle.dump(y, file)
        else:
            print("UNKNOWN dataset")
            exit(2)
    return x, y

def load_train_dev_test(dataset_name):
    formated_train_xfile = "../data/formated_data/"+dataset_name+"/train_x.pkl"
    formated_train_yfile = "../data/formated_data/"+dataset_name+"/train_y.pkl"
    formated_dev_xfile = "../data/formated_data/"+dataset_name+"/dev_x.pkl"
    formated_dev_yfile = "../data/formated_data/"+dataset_name+"/dev_y.pkl"
    formated_test_xfile = "../data/formated_data/"+dataset_name+"/test_x.pkl"
    formated_test_yfile = "../data/formated_data/"+dataset_name+"/test_y.pkl"
    with open(formated_train_xfile, "rb") as file:
        train_x = pickle.load(file)
    with open(formated_train_yfile, "rb") as file:
        train_y = pickle.load(file)
    with open(formated_dev_xfile, "rb") as file:
        dev_x = pickle.load(file)
    with open(formated_dev_yfile, "rb") as file:
        dev_y = pickle.load(file)
    with open(formated_test_xfile, "rb") as file:
        test_x = pickle.load(file)
    with open(formated_test_yfile, "rb") as file:
        test_y = pickle.load(file)
    return train_x, train_y, dev_x, dev_y, test_x, test_y

def get_train_dev_test(x, y, dataset_name):
    formated_train_xfile = "../data/formated_data/"+dataset_name+"/train_x.pkl"
    formated_train_yfile = "../data/formated_data/"+dataset_name+"/train_y.pkl"
    formated_dev_xfile = "../data/formated_data/"+dataset_name+"/dev_x.pkl"
    formated_dev_yfile = "../data/formated_data/"+dataset_name+"/dev_y.pkl"
    formated_test_xfile = "../data/formated_data/"+dataset_name+"/test_x.pkl"
    formated_test_yfile = "../data/formated_data/"+dataset_name+"/test_y.pkl"
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    dev_test_len = int(len(x_dev)/2)
    x_test, y_test = x_dev[:dev_test_len], y_dev[:dev_test_len]
    x_dev, y_dev = x_dev[dev_test_len:], y_dev[dev_test_len:]
    
    with open(formated_train_xfile, "wb") as file:
        pickle.dump(x_train, file)
    with open(formated_train_yfile, "wb") as file:
        pickle.dump(y_train, file)
    with open(formated_dev_xfile, "wb") as file:
        pickle.dump(x_dev, file)
    with open(formated_dev_yfile, "wb") as file:
        pickle.dump(y_dev, file)
    with open(formated_test_xfile, "wb") as file:
        pickle.dump(x_test, file)
    with open(formated_test_yfile, "wb") as file:
        pickle.dump(y_test, file)
    return x_train, y_train, x_dev, y_dev, x_test, y_test

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def preprocess_mr(pos_data_file, neg_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(pos_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def float_to_int(float_str):
    float_str = float(float_str)
    if float_str <= 0.2:
        return [1,0,0,0,0]
    elif float_str <= 0.4:
        return [0,1,0,0,0]
    elif float_str <= 0.6:
        return [0,0,1,0,0]
    elif float_str <= 0.8:
        return [0,0,0,1,0]
    else:
        return [0,0,0,0,1]

def preprocess_sst(file_name):
    x_text = list(open(file_name, "r").readlines())
    y = [float_to_int(x.strip().split("\t")[1]) for x in x_text]
    x_text = [x.strip().split("\t")[0] for x in x_text]
    x_text = [clean_str(sent) for sent in x_text]
    return x_text, y

def preprocess_sst1(dataset_name):
    if os.path.isdir("../data/formated_data/"+dataset_name+"/") == False:
        os.mkdir("../data/formated_data/"+dataset_name+"/")
    x, y = preprocess_sst("../data/sst1/train.txt")
    with open("../data/formated_data/"+dataset_name+"/train_x.pkl", "wb") as file:
        pickle.dump(x, file)
    with open("../data/formated_data/"+dataset_name+"/train_y.pkl", "wb") as file:
        pickle.dump(y, file)
    x, y = preprocess_sst("../data/sst1/dev.txt")
    with open("../data/formated_data/"+dataset_name+"/dev_x.pkl", "wb") as file:
        pickle.dump(x, file)
    with open("../data/formated_data/"+dataset_name+"/dev_y.pkl", "wb") as file:
        pickle.dump(y, file)
    x, y = preprocess_sst("../data/sst1/test.txt")
    with open("../data/formated_data/"+dataset_name+"/test_x.pkl", "wb") as file:
        pickle.dump(x, file)
    with open("../data/formated_data/"+dataset_name+"/test_y.pkl", "wb") as file:
        pickle.dump(y, file)

def shuffle_data(x, y):
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    # print(y)
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled, y_shuffled

def load_company_data_x(filename):
    with open(filename, "rb") as file:
        train_x = pickle.load(file)
        train_x = [" ".join(list(x)) for x in train_x]
        # print(train_x)
        return train_x

def load_company_data_y(filename):
    with open(filename, "rb") as file:
        y = pickle.load(file)
        return y

def split_data(flag):
    x = load_company_data_x(company_x)
    if flag == "text":
        y = load_company_data_y(text_y)
    elif flag == "sen3":
        y = load_company_data_y(sen3_y)
    elif flag == "sen5":
        y = load_company_data_y(sen5_y)
    np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(0.2 * float(len(y)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    dev_test_len = int(len(x_dev)/2)
    x_test, y_test = x_dev[:dev_test_len], y_dev[:dev_test_len]
    x_dev, y_dev = x_dev[dev_test_len:], y_dev[dev_test_len:]
    with open("../data/formated_data/company/"+flag+"/train_x.pkl", "wb") as file:
        pickle.dump(x_train, file)
    with open("../data/formated_data/company/"+flag+"/train_y.pkl", "wb") as file:
        pickle.dump(y_train, file)
    with open("../data/formated_data/company/"+flag+"/dev_x.pkl", "wb") as file:
        pickle.dump(x_dev, file)
    with open("../data/formated_data/company/"+flag+"/dev_y.pkl", "wb") as file:
        pickle.dump(y_dev, file)
    with open("../data/formated_data/company/"+flag+"/test_x.pkl", "wb") as file:
        pickle.dump(x_test, file)
    with open("../data/formated_data/company/"+flag+"/test_y.pkl", "wb") as file:
        pickle.dump(y_test, file)

def load_company_data(flag):
    formated_train_xfile = "../data/formated_data/company/"+flag+"/train_x.pkl"
    formated_train_yfile = "../data/formated_data/company/"+flag+"/train_y.pkl"
    formated_dev_xfile = "../data/formated_data/company/"+flag+"/dev_x.pkl"
    formated_dev_yfile = "../data/formated_data/company/"+flag+"/dev_y.pkl"
    formated_test_xfile = "../data/formated_data/company/"+flag+"/test_x.pkl"
    formated_test_yfile = "../data/formated_data/company/"+flag+"/test_y.pkl"
    with open(formated_train_xfile, "rb") as file:
        train_x = pickle.load(file)
    with open(formated_train_yfile, "rb") as file:
        train_y = pickle.load(file)
    with open(formated_dev_xfile, "rb") as file:
        dev_x = pickle.load(file)
    with open(formated_dev_yfile, "rb") as file:
        dev_y = pickle.load(file)
    with open(formated_test_xfile, "rb") as file:
        test_x = pickle.load(file)
    with open(formated_test_yfile, "rb") as file:
        test_y = pickle.load(file)
    return train_x, train_y, dev_x, dev_y, test_x, test_y