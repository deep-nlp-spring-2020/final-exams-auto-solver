import pandas as pd
import re
import torch

from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction, XLMRobertaTokenizer,XLMRobertaModel
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import ToktokTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class BertEmbedding:
    def __init__(self, bert_name="bert-base-multilingual-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.model = BertModel.from_pretrained(bert_name, output_attentions = True, output_hidden_states = True)
        self.model.cuda()

    def sentence_embedding(self, text, text_list):
        embeddings = []
        for variant in text_list:
           ts = self.tokenizer.encode_plus(text,variant, add_special_tokens=True)
           th_inputs = torch.tensor([ts['input_ids']]).device('cuda')
           th_type = torch.tensor([ts['token_type_ids']])
           try:
               with torch.no_grad():
                    outputs = self.model(th_inputs, token_type_ids=th_type)
                    import torch.nn.functional as F
                    outputs = F.softmax(outputs[0])
               embeddings.append(outputs[0].tolist())
           except BaseException as e:
               print(e)

        return embeddings

    def sentence_variance_embedding(self, text_list):
        embeddings = []
        for text in text_list:
            text = text.replace('1)', '')
            text = text.replace('2)', '')
            text = text.replace('3)', '')
            text = text.replace('4)', '')
            text = text.replace('5)', '')
            token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
            segments_ids, indexed_tokens = (
                [1] * len(token_list),
                self.tokenizer.convert_tokens_to_ids(token_list),
            )
            segments_tensors, tokens_tensor = (
                torch.tensor([segments_ids]),
                torch.tensor([indexed_tokens]),
            )
            with torch.no_grad():
                encoded_layers, pooled_layers, hs, attentions = self.model(tokens_tensor.cuda(), segments_tensors.cuda())
            sent_embedding = torch.mean(hs[6], 1)
            embeddings.append(sent_embedding.to('cpu'))
        return embeddings


class MainTextIdea:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.toktok = ToktokTokenizer()
        self.bert = BertEmbedding()

    def compare_text_with_variants(self, text, variants):
        variant_vectors = self.bert.sentence_embedding(text, variants)
        predicts = pd.DataFrame(variant_vectors, columns=['col1', 'col2'])
        predicts_new = predicts.sort_values(by=['col1'], ascending=False)[:2]
        indexes = predicts_new.index.tolist()
        return sorted([str(i + 1) for i in indexes])

    def compare_variants(self, variants):
        variant_vectors = self.bert.sentence_variance_embedding(variants)
        predicts = []
        for i in range(0, len(variant_vectors)):
            for j in range(i + 1, len(variant_vectors)):
                sim = cosine_similarity(
                    variant_vectors[i].reshape(1, -1), variant_vectors[j].reshape(1, -1)
                ).flatten()[0]
                predicts.append(pd.DataFrame({"sim": sim, "i": i, "j": j}, index=[1]))
        predicts = pd.concat(predicts)
        indexes = predicts[predicts.sim == predicts.sim.max()][["i", "j"]].values[0]
        return sorted([str(i + 1) for i in indexes])

    def process_task(self, task):
        first_phrase, task_text = re.split(r"\(*1\)", task["text"])[:2]
        variants = [t["text"] for t in task["question"]["choices"]]
        text, task = "", ""
        if "Укажите" in task_text:
            text, task = re.split("Укажите ", task_text)
            task = "Укажите " + task
        elif "Укажите" in first_phrase:
            text, task = task_text, first_phrase
        return text, task, variants

    def predict_from_model(self, task):
        text, task, variants = self.process_task(task)
        result = self.compare_variants(variants)
        return result