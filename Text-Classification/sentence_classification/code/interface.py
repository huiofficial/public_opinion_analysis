import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import data_preprocess
import wrapper

if __name__ == "__main__":
    # wrapper.text_classification()
    # wrapper.sen3_classification()
    assert len(sys.argv) == 2
    if sys.argv[1] == "text":
        wrapper.text_classification()
    elif sys.argv[1] == "sen3":
        wrapper.sen3_classification()
    elif sys.argv[1] == "sen5":
        wrapper.sen5_classification()
    else:
        raise NotImplementedError(sys.argv[1]+" is not implemented")
    # data_preprocess.split_data("text")
    # data_preprocess.split_data("sen3")
    # data_preprocess.split_data("sen5")
    # train_x, train_y, dev_x, dev_y, test_x, test_y = data_preprocess.load_company_data("text")
    # print(len(train_x))
    # print(len(dev_x))
    # print(len(test_x))
    # max_document_length = max([len(i.split(" ")) for i in train_x])
    # print(max_document_length)
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # print(train_x[0])
    # vocab_processor.fit(train_x)
    # train_x = np.array(list(vocab_processor.transform(train_x)))
    # print(train_x[0])
    # while True:
    #     s = input("input:\n")
    #     s = data_preprocess.clean_str(s)
    #     print(s)
    #     s = np.array(list(vocab_processor.transform([" ".join(list(s))])))
    #     print(s)