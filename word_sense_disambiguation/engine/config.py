from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig, AutoTokenizer, AutoModelWithLMHead,AutoModel,BertForTokenClassification
import numpy as np
MYPATH_TRAIN = '../public_set/train/'
MYPATH_TEST = '../public_set/test_no_ids/'
MYPATH_VAL = '../public_set/test/'

PATH_TO_TEST_DATA_CONCATED = '../data/test/test.json'

# LAYERS_TO_RETRIEVE = [9,11]

METHODS_TO_EMBED = ['concat_last_n_layers_cls', 'concat_last_n_layers_over_all']

METHOD_TO_EMBED = METHODS_TO_EMBED[1]

LAYERS_TO_EMBED = np.arange(0,12)

#bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
#model = BertModel.from_pretrained('bert-base-multilingual-cased')

bert_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

model = BertForTokenClassification.from_pretrained(
     'DeepPavlov/rubert-base-cased',
     output_attentions = True,
     output_hidden_states = True
        )