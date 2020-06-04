from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import re
import operator
import random
import pymorphy2
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from scipy.spatial.distance import cosine
from razdel import tokenize, sentenize
import warnings
import config
from os import listdir
from os.path import isfile, join
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertConfig, AutoTokenizer, AutoModelWithLMHead,AutoModel,BertForTokenClassification

'''
get all files from data directory to list
'''
def list_files_in_dir(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

class TextTokenizer:
    def tokenize(self, text):
        return [_.text for _ in tokenize(text)]

    def sentenize(self, text):
        return [_.text for _ in sentenize(text)]

    
def sentence_split(text):
    return re.split(r'\(\n*\d+\n*\)', text)
    
    
def make_data_(mypath, dump = True, name_dump = 'train.json'):
    file_names = list_files_in_dir(mypath)
    all_data = []
    for file in file_names:
        data = json.load(open(mypath + file, 'r', encoding = 'utf-8'))
        all_data.append(data[2])
    if dump == True:
        with open(name_dump, 'w') as f:
            json.dump(all_data, f)
        return 0
    else:
        return all_data    
    
def sent_token_embedding(model, bert_tokenizer, instance, layers_to_retrieve,method = 'concat_last_n_layers_cls'):
    tokenized_word = bert_tokenizer.tokenize("[CLS] " + instance + " [SEP]")
    
    attention_mask, input_ids = [1] * len(tokenized_word), bert_tokenizer.convert_tokens_to_ids(tokenized_word)
    attention_mask_ = torch.tensor([attention_mask], dtype = torch.long)
    
    input_ids_ = torch.tensor([input_ids], dtype = torch.long)

    with torch.no_grad():
        try:
            embeddings, _ = model(input_ids_, attention_mask_)
        except ValueError:
            _, embeddings,_ = model(input_ids_, attention_mask_)
            embeddings = list(embeddings)[:12]
    last_layers_embeddings = []
    for layer in layers_to_retrieve:
        last_layers_embeddings.append(embeddings[layer])
    embs_last = torch.stack(tuple(last_layers_embeddings), axis = 0).squeeze(1).permute(1,0,2)
    
    '''
    it's possible to retrieve only one layer by calling layers_to_retrieve = [9] if we want 9th layer
    '''
        
    if method == 'concat_last_n_layers_cls':

        return torch.cat([embs_last[0][i] for i in range(len(last_layers_embeddings))], dim=0).numpy()
    
    elif method == 'concat_last_n_layers_over_all':

        return torch.cat([embs_last.mean(0)[i] for i in range(len(last_layers_embeddings))], dim=0).numpy()
        
        
        

def embed_target_sentence(model, text, target_word):
    sentences = tokenizer.sentenize(text)
    for sentence in sentences:
        words = tokenizer.tokenize(sentence)
        lemmas = [morph.parse(word) for word in
              words]
        for num_de_lem in range(len(lemmas)):
            if morph.parse(target_word)[0].normal_form in [el.normal_form for el in lemmas[num_de_lem]]:
                return sent_token_embedding(model, bert_tokenizer, sentence, layers_to_embed,method_to_embed)        
        

        

def contextualized_word_embedding(model, text, target_word, layers_to_retrieve = [11]):
    sentences = tokenizer.sentenize(text)
    target_sentence = None
    index_of_word_in_sentence = -1
    for sentence in sentences:
        words = tokenizer.tokenize(sentence)
        lemmas = [morph.parse(word) for word in
              words]
        
        for num_de_lem in range(len(lemmas)):
            if morph.parse(target_word)[0].normal_form in [el.normal_form for el in lemmas[num_de_lem]]:
                index_of_word_in_sentence = num_de_lem
                target_sentence = sentence
                break
    
    if index_of_word_in_sentence == -1:
        target_sentence = ' '.join(sentences)
        start_index = 0
        end_index = len(tokenizer.tokenize(target_sentence)) - 2
        print("couldn't find target word in sentences")
        print('target sentence: {0} \n word : {1}'.format(target_sentence, target_word))
        print()
    
    words = tokenizer.tokenize(target_sentence)
    words.insert(index_of_word_in_sentence, '|')
    words.insert(index_of_word_in_sentence +2 , '|')
    
    target_sentence_tokenized = bert_tokenizer.tokenize("[CLS] " + ' '.join(words) + " [SEP]")
    
    start_index = target_sentence_tokenized.index('|')
    target_sentence_tokenized.pop(start_index)
    end_index = target_sentence_tokenized.index('|')
    target_sentence_tokenized.pop(end_index)
    
    attention_mask, input_ids = [1] * len(target_sentence_tokenized), bert_tokenizer.convert_tokens_to_ids(target_sentence_tokenized)
    attention_mask_ = torch.tensor([attention_mask], dtype = torch.long)
    
    input_ids_ = torch.tensor([input_ids], dtype = torch.long)

    with torch.no_grad():
        try:
            embeddings, _ = model(input_ids_, attention_mask_)
        except ValueError:
            _, embeddings,_ = model(input_ids_, attention_mask_)
            embeddings = list(embeddings)[:12]
    
    last_layers_embeddings = []
    for layer in layers_to_retrieve:
        last_layers_embeddings.append(embeddings[layer][:,start_index:end_index, :]) #take only word piece embeddings, which correspond to 
    
    embs_last = torch.stack(tuple(last_layers_embeddings), axis = 0).squeeze(1).permute(1,0,2)
    
    return torch.cat([embs_last.mean(0)[i] for i in range(len(last_layers_embeddings))], dim=0).numpy()
    
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
    variants = [re.sub('\d+[.)]', '', variant) for variant in variants]
    return re.sub('Прочитайте фрагмент.*', '', text), file, variants, word.lower()

def clean_text_additional(text):
    text = re.sub(r'\(.*?\)', "", text)
    #text = re.sub(r"\<.*?\>", '', text)
    return text
def cosine_argmax(embedded_text, embedded_list_of_variants):
    lst_cosine = []
    for variant_num in range(len(embedded_list_of_variants)):
        lst_cosine.append(cosine_similarity(embedded_list_of_variants[variant_num].reshape(1, -1), embedded_text.reshape(1,-1))[0][0])
    
    return np.argmax(lst_cosine) + 1


if __name__ == '__main__':
    morph = pymorphy2.MorphAnalyzer()
    tokenizer = TextTokenizer()
    bert_tokenizer = config.bert_tokenizer
    model = config.model
    model.eval()
    correct = []
    predicted = []
    method_to_embed = config.METHOD_TO_EMBED
    layers_to_embed = config.LAYERS_TO_EMBED
    
    test_df = json.load(open(config.PATH_TO_TEST_DATA_CONCATED, 'r', encoding = 'utf-8'))

    for file in test_df:
        text, task, variants, word  = preprocess_file(file) #need to remove (\d) in order to do correct sentenize
        text = clean_text_additional(text)
        target = get_correct_label(file)
        
        embedded_text = contextualized_word_embedding(model, text, word, layers_to_retrieve = layers_to_embed)
        
        embeddings_cases = []
        for j in range(len(variants)):
            embeddings_cases.append(sent_token_embedding(model, bert_tokenizer, variants[j], layers_to_retrieve = layers_to_embed,
                              method = method_to_embed))



        
        predicted_var = cosine_argmax(embedded_text, embeddings_cases)

        predicted.append(predicted_var)
        correct.append(target)

    print(accuracy_score(correct, predicted))

