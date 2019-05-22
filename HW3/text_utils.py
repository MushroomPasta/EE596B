import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import io
"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches


"""
class TextLoader():
    def __init__(self):
        pass
    def load_words(self,filename):
        filename = 'shakespeare.txt'
        with io.open(filename, encoding='utf-8') as f:
            self.text = f.read().lower()
        print('corpus length:', len(self.text))

        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # cut the text in semi-redundant sequences of maxlen characters
        self.maxlen = 40
        self.step = 3
        self.sentences = []
        self.next_chars = []
        for i in range(0, len(self.text) - self.maxlen, self.step):
            self.sentences.append(self.text[i: i + self.maxlen])
            self.next_chars.append(self.text[i + self.maxlen])
        print('nb sequences:', len(self.sentences))

        print('Vectorization...')
        x = np.zeros((len(self.sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[self.next_chars[i]]] = 1
        return self.maxlen, self.text, self.indices_char,self.char_indices,self.chars,x,y
#        self.char = []
#        self.text = open(filename,'r').read()
#        self.strings = re.findall(r"[\w']+", self.text)
##        self.strings = self.text.read().split(':|;|,|--|?|!|\n| ')
#        for i in range(len(self.strings)):
##            if not self.strings[i][-1].isalpha():
##                self.strings[i] = self.strings[i][:-1]
#            if self.strings[i] not in self.char:
#                self.char.append(self.strings[i])
#        self.vocab_size = len(self.char)
#        self.vocab = {}
#        for i in range(self.vocab_size):
#            self.vocab[self.char[i]] = i
#        print(len(self.char),'unique words')
#        return tuple(self.char),self.vocab
        
#a = TextLoader()
#b,c = a.load_words('shakespeare.txt')



        
        
        
        