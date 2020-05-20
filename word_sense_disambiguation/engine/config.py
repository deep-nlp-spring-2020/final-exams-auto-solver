MYPATH_TRAIN = '../public_set/train/'
MYPATH_TEST = '../public_set/test_no_ids/'
MYPATH_VAL = '../public_set/test/'

PATH_TO_TEST_DATA_CONCATED = '../data/test/test.json'

LAYERS_TO_RETRIEVE = [-1, -2, -3, -4]

METHODS_TO_EMBED = ['concat_last_n_layers_cls', 'concat_last_n_layers_over_all']

METHOD_TO_EMBED = METHODS_TO_EMBED[1]

LAYERS_TO_EMBED = [9, 11]