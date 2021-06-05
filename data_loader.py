import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter


class Data_Loader:
    def __init__(self, options):
        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        padid = str(options['padid'])

        split_tip = ','

        max_document_length = max([len(x.split(split_tip)) for x in positive_examples])

        new_positive_examples = []
        for x in positive_examples:
            x = x.strip()
            x_list = x.split(split_tip)
            x_len = len(x_list)
            if x_len != max_document_length:
                padlen = max_document_length - x_len
                x_list = padlen * [padid] + x_list
                x = split_tip.join(x_list)
            new_positive_examples.append(x)

        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(new_positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping