from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from transformers import BertModel, BertTokenizer, BertConfig
import re
import operator
import random
import pymorphy2
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from scipy.spatial.distance import cosine
from razdel import tokenize, sentenize
from transformers import BertForSequenceClassification
import warnings
import config

'''
get all files from data directory to list
'''
def list_files_in_dir(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

class Tokenizer:
    def __init__(self):
        pass
    def word_tokenizer(self, text):
        return [_.text for _ in list(tokenize(text))]
    
    def sentence_tokenizer(self, text):
        return [_.text for _ in list(sentenize(text))]
    

def bert_token_func():
    return BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)

def sent_token_embedding(model, tokenizer, instance, layers_to_retrieve,method = 'concat_last_n_layers_cls'):

    tokenized_word = tokenizer.tokenize(instance)
    inputs = tokenizer.encode_plus(
        tokenized_word,
        None,
        add_special_tokens=True
    )
    attention_mask = torch.tensor([inputs['attention_mask']], dtype = torch.long)
    #token_type_ids = torch.tensor(inputs['token_type_ids'], dtype = torch.long)
    input_ids = torch.tensor([inputs['input_ids']], dtype = torch.long)

    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)
    if method == 'concat_last_n_layers_cls':
        '''
        it's possible to retrieve only one layer by calling layers_to_retrieve = [11] if we want 11th layer
        '''
        
        last_layers_embeddings = []
        for layer in layers_to_retrieve:
            last_layers_embeddings.append(embeddings[2][layer])
        embs_last = torch.stack(tuple(last_layers_embeddings), axis = 0).squeeze(1).permute(1,0,2)
        return np.expand_dims(torch.cat([embs_last[0][i] for i in range(len(last_layers_embeddings))], dim=0).numpy(), axis =1)
    
    elif method == 'concat_last_n_layers_over_all':
        last_layers_embeddings = []
        for layer in layers_to_retrieve:
            last_layers_embeddings.append(embeddings[2][layer])
        embs_last = torch.stack(tuple(last_layers_embeddings), axis = 0).squeeze(1).permute(1,0,2)
        return np.expand_dims(torch.cat([embs_last.mean(0)[i] for i in range(len(last_layers_embeddings))], dim=0).numpy(), axis = 1)
        
        
        
def load_model(model_name ='bert-base-multilingual-cased'):
    
    model = BertModel.from_pretrained(
    model_name, output_hidden_states=True)
    
    return model
        
def get_correct_label(file):
    return int(file['solution']['correct'][0])


def preprocess_file(file):
    try:
        question, file_text = re.split(r'\(\n*1\n*\)', file['text'])
        
    except ValueError:
        question, file_text = ' '.join(re.split(r'\(\n*1\n*\)', file['text'])[:-1]), \
                                re.split(r'\(\n*1\n*\)', file['text'])[-1]
    variants = [t['text'] for t in file['question']['choices']]
    text = ''
    file = ''
    word = ''
    if 'Определите' in file_text:
        text, file = re.split('Определите', file_text)
        file = 'Определите ' + file
        word = re.split('\.', re.split('значения слова ', text)[1])[0]
    elif 'Определите' in question:
        text, file = file_text, question
        word = re.split('\.', re.split('значения слова ', file)[1])[0]
    return text, file, variants, word



def cosine_argmax(embedded_text, embedded_list_of_variants):
    lst_cosine = []
    for variant_num in range(len(embedded_list_of_variants)):
        lst_cosine.append(1 - cosine(embedded_list_of_variants[variant_num], embedded_text))
    
    return np.argmax(lst_cosine)


if __name__ == '__main__':
    mypath_train = config.MYPATH_TRAIN
    mypath_test = config.MYPATH_TEST
    mypath_val = config.MYPATH_VAL
    file_names = list_files_in_dir(mypath_train)
    
    layers_to_retrieve = config.LAYERS_TO_RETRIEVE
    model = load_model()
    cust_tokenizer = Tokenizer()
    bert_tokenizer = bert_token_func()
    
    correct = []
    predicted = []
    method_to_embed = config.METHODS_TO_EMBED[0]
    for file in file_names:
        data = json.load(open(mypath_train + file, 'r', encoding = 'utf-8'))
        text, task, variants, word  = preprocess_file(data[2])
        target = get_correct_label(data[2])
        layers_to_embed = [-1,-2,-3,-4]
        embeddings_cases = []
        for j in range(len(variants)):
            embeddings_cases.append(sent_token_embedding(model, bert_tokenizer, variants[j], layers_to_retrieve = layers_to_embed,
                              method = method_to_embed))
        
        
        embedded_text = sent_token_embedding(model, bert_tokenizer, text, layers_to_retrieve = layers_to_embed,
                              method = method_to_embed)
        

        
        predicted_var = cosine_argmax(embedded_text, embeddings_cases)
        
        predicted.append(predicted_var)
        correct.append(target)
                    
    print(accuracy_score(correct, predicted))
    
        