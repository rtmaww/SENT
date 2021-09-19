import json
import random
random.seed(1)
import numpy as np
import torch
from copy import deepcopy
import pickle
import copy
from collections import defaultdict
import sys
# sys.path.append('./')
from .GloveTokenizer import WordTokenizer

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, head_span, tail_span, token_masks):
        self.input_ids = input_ids
        self.head_span = head_span
        self.tail_span = tail_span
        self.token_masks = token_masks


class Instance(object):

    def __init__(self, words, relation, head, tail, headpos, tailpos, headtype, tailtype, ner=None, is_noise=None):
        self.words = words
        self.relation = relation
        self.head = head
        self.tail = tail
        self.headpos = headpos
        self.tailpos = tailpos
        self.headtype = headtype
        self.tailtype = tailtype
        self.d_rel = ""
        self.ner = ner
        self.is_noise = is_noise

class Data():
    def __init__(self, args, mode='train'):

        if mode == 'train':
            data_file = args.train_data_file
        elif mode == 'test':
            data_file = args.test_data_file
        elif mode == 'dev':
            data_file = args.dev_data_file
        elif mode == 'test_noise':
            data_file = args.test_noise_file

        self.dataset = "nyt"

        rel2id_file = args.rel2id_file

        self.max_len = args.max_len

        self.tokenizer = WordTokenizer(args.vocab_file)

        self.use_noise_label = args.noise_label

        self.create_label_dict()
        self.facts = defaultdict(set)
        # Load and preprocess data
        print('Data Loading!-----')

        if self.use_noise_label and mode == 'test_noise':
            data = self.load_data_nyt_arnor_ner_noise(data_file, rel2id_file)
        else:
            data = self.load_data_nyt_arnor_ner(data_file, rel2id_file)
        ori_data_len = len(data)

        print('Data Loaded!-----')

        print('Data Preprocessing!-----')

        features = self.preprocess(data)
        print('Data Preprocessed!-----')

        print('Processed Data List Creating!----')
        self.processed_data = []
        delete_index = []
        self.rel_num = defaultdict(int)
        for _,(item, feature) in enumerate(zip(data, features)):

            if feature is None:
                delete_index.append(_)
                continue
            temp_item = {}
            temp_item['input_ids'] = feature.input_ids
            temp_item['e1_begin'] = feature.head_span[0]
            temp_item['e1_end'] = feature.head_span[1]
            temp_item['e2_begin'] = feature.tail_span[0]
            temp_item['e2_end'] = feature.tail_span[1]
            if not item.relation:
                delete_index.append(_)
                continue
            temp_item['rel'] = item.relation
            temp_item['ori_sentence'] = item.words
            temp_item['token_masks'] = feature.token_masks
            temp_item['bag_name'] = (item.head, item.tail, item.relation)
            temp_item['ner'] = item.ner
            if self.use_noise_label:
                temp_item['is_noise'] = item.is_noise
            self.rel_num[item.relation] += 1

            self.processed_data.append(temp_item)
        print('Processed Data List Created!----')
        print('Processed data has {} instances'.format(len(self.processed_data)))

        for rel, num in self.rel_num.items():
            print("{}: {}".format(rel, num))
        # self.batchify()


    def load_predenoise_labels(self, path, describe=""):
        with open(path+describe+'_labels.txt', 'r') as f:
            labels = json.load(f)
        return  labels

    def load_data_nyt_arnor_ner(self, data_file, rel2id_file, load_ner=True):

        self.create_label_dict(rel2id_file)
        if load_ner:
            self.create_ner_dict()
        with open(data_file, 'r') as infile:
            data = json.load(infile)

        instances = []

        for item in data:
            words = item["sentence"].split(" ")
            if len(words) > self.max_len:
                continue
            relation = item['relation']
            if relation == 'None':
                relation = 'NA'

            head = item['head']['word']
            tail = item['tail']['word']
            if relation != "NA":
                self.facts[(head, tail)].add(relation)


            try:
                head_list = head.split()
                pos = -1
                while True:
                    pos = words.index(head_list[0], pos + 1)
                    if " ".join(words[pos:pos + len(head_list)]) == head:
                        head_pos = (pos, pos + len(head_list)-1)
                        break

                tail_list = tail.split()
                pos = -1
                while True:
                    pos = words.index(tail_list[0], pos + 1)
                    if " ".join(words[pos:pos + len(tail_list)]) == tail:
                        tail_pos = (pos, pos + len(tail_list)-1)
                        break
            except:
                continue

            head_type = item['head']['type']
            tail_type = item['tail']['type']
            if load_ner:
                ner = [self.ner2id[i] for i in item['stanford_ner']]
            else:
                ner = None
            instances.append(Instance(words, relation, head, tail, head_pos, tail_pos, head_type, tail_type, ner))

        print('Original data has {} instances'.format(len(instances)))

        print("***** print examples ******")
        for ins in instances[:5]:
            print("words: {}, head: {}, head_pos: {}, tail: {}, tail_pos: {}, relation: {}, d_rel: {}, ner: {}"
                  .format(" ".join(ins.words), ins.head, str(ins.headpos), ins.tail, str(ins.tailpos), ins.relation, ins.d_rel, ins.ner))


        return instances

    def load_data_nyt_arnor_ner_noise(self, data_file, rel2id_file, load_ner=True):


        self.create_label_dict(rel2id_file)
        if load_ner:
            self.create_ner_dict()
        with open(data_file, 'r') as infile:
            data = json.load(infile)

        instances = []

        for item in data:
            words = item["sentence"].split(" ")
            if len(words) > self.max_len:
                continue
            relation = item['relation']
            if relation == 'None':
                relation = 'NA'

            head = item['head']['word']
            tail = item['tail']['word']

            if relation != "NA":
                self.facts[(head, tail)].add(relation)

            try:
                head_list = head.split()
                pos = -1
                while True:
                    pos = words.index(head_list[0], pos + 1)
                    if " ".join(words[pos:pos + len(head_list)]) == head:
                        head_pos = (pos, pos + len(head_list)-1)
                        break

                tail_list = tail.split()
                pos = -1
                while True:
                    pos = words.index(tail_list[0], pos + 1)
                    if " ".join(words[pos:pos + len(tail_list)]) == tail:
                        tail_pos = (pos, pos + len(tail_list)-1)
                        break
            except:
                continue

            head_type = item['head']['type']
            tail_type = item['tail']['type']
            if load_ner:
                ner = [self.ner2id[i] for i in item['stanford_ner']]
            else:
                ner = None
            is_noise = item["is_noise"]
            instances.append(Instance(words, relation, head, tail, head_pos, tail_pos, head_type, tail_type, ner, is_noise))

        print('Original data has {} instances'.format(len(instances)))

        print("***** print examples ******")
        for ins in instances[:5]:
            print("words: {}, head: {}, head_pos: {}, tail: {}, tail_pos: {}, relation: {}, d_rel: {}, ner: {}"
                  .format(" ".join(ins.words), ins.head, str(ins.headpos), ins.tail, str(ins.tailpos), ins.relation, ins.d_rel, ins.ner))


        return instances

    def create_ner_dict(self):
        file = "/home/mrt/data/arnor_nyt/1.0/ner2id_.json"
        with open(file, 'r') as f:
            self.ner2id = json.load(f)

    def get_label_num(self):
        return len(self.relId2labelId)

    def preprocess(self, data, token_mask_id=0):
        features = []
        unk = 0
        for idx, item in enumerate(data):
            tokens = item.words
            # tokens = self.tokenizer.tokenize(item)
            if len(tokens) > self.max_len:
                features.append(None)
                continue
            input_ids, unk_num = self.tokenizer.convert_tokens_to_ids(tokens, self.max_len,
                                                             self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]'], uncased=True)
            head_span = [item.headpos[0], item.headpos[-1]+1]
            tail_span = [item.tailpos[0], item.tailpos[-1]+1]
            token_masks = [1] * len(input_ids)

            if idx < 5:
                print("*** Example ***")
                print("tokens: {}".format(" ".join(tokens)))
                print("E1 position:({}, {}), E2 position:({}, {})".format(head_span[0], head_span[1],
                                                                          tail_span[0], tail_span[1]))
                print("token mask: {}".format(str(token_masks)))
                print('input ids: {}'.format(str(input_ids)))

            features.append(
                InputFeatures(input_ids=input_ids,
                              head_span=head_span,
                              tail_span=tail_span,
                              token_masks=token_masks))
            unk += unk_num
        print("Convert token to vocab id, unk token num: {}".format(unk))
        return features

    def get_vocab_size(self):
        return len(self.tokenizer.vocab)

    def create_label_dict(self, file=None):
        if file is None:
            self.relId2labelId = LABEL_TO_ID
            self.labelId2rel = {v:k for k,v in self.relId2labelId.items()}
        else:
            with open(file, 'r') as f:
                line = json.load(f)
            self.relId2labelId = line
            self.labelId2rel = {v: k for k, v in self.relId2labelId.items()}

    def get_labels(self):
        return self.labels

    def id2rel(self, id):
        return self.labelId2rel.get(id, None)

    def rel2id(self, rel):
        return self.relId2labelId.get(rel, None)

    def get_label_num(self):
        return len(self.relId2labelId)

    # data_processing functions
    def posnum_to_posarray(self, posbegin, posend):
        if (posend < posbegin):
            posend = posbegin
        array1 = np.arange(0,posbegin) - posbegin
        array2 = np.zeros(posend-posbegin,dtype=np.int32)
        array3 = np.arange(posend,self.max_len) - posend
        posarray = np.append(np.append(array1, array2), array3) + self.max_len
        return posarray


    def batchify(self, noise_label=False):

        batch_data = []
        PAD = self.tokenizer.vocab['[PAD]']
        ner_PAD = self.ner2id['[PAD]']
        for i, item in enumerate(self.processed_data):
            # try:
            padding_size = self.max_len - len(item['input_ids'])
            ori_token_masks = torch.LongTensor(item['token_masks'] + [0] * padding_size)

            head_masks = torch.zeros(len(item['input_ids'])+padding_size).long()
            head_masks[item['e1_begin']:item['e1_end']] = 1

            head_masks = head_masks * ori_token_masks
            tail_masks = torch.zeros(len(item['input_ids'])+padding_size).long()
            tail_masks[item['e2_begin']:item['e2_end']] = 1
            tail_masks = tail_masks * ori_token_masks

            # padded_idx_data = np.zeros([5, self.max_len], dtype=np.int32)
            head_pos = torch.LongTensor(self.posnum_to_posarray(item['e1_begin'], item['e1_end']-1))
            tail_pos = torch.LongTensor(self.posnum_to_posarray(item['e2_begin'], item['e2_end']-1))

            try:
                assert head_pos.size(0) == self.max_len
            except:
                print(item['e1_begin'], item['e1_end'])


            input_ids = torch.LongTensor(item['input_ids']+[PAD]*padding_size)
            input_masks = torch.LongTensor([1]*len(item['input_ids'])+[0]*padding_size)  ###sentence mask

            labels = torch.LongTensor([self.relId2labelId[item['rel']]])

            ner_labels = torch.LongTensor(item['ner'] + [ner_PAD] * padding_size)
            batch_data.append(
                [head_pos, tail_pos, input_ids, input_masks, ori_token_masks, head_masks, tail_masks, ner_labels,
                 labels])

            if noise_label:
                is_noise = item["is_noise"]
                batch_data[-1].append(is_noise)

        return batch_data


    def dumpData(self, save_path):
        with open(save_path, 'wb'):
            pickle.dump(self,save_path)
