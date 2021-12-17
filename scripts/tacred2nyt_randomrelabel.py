import json
from collections import defaultdict
import random
import os

random.seed(1)
# TRAIN = True

mode_list = ["train", "dev", "test"]

file_path = "../data/tacred/data/json/" ## tacred data path

for mode in mode_list:

    input_file = os.path.join(file_path,"{}.json".format(mode))
    output_file = os.path.join(file_path,"{}_random0.3.json".format(mode))
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    converted_data = []
    label2id = defaultdict(int)
    pairdict = {}
    labeldict = defaultdict(int)
    ner2id = {'[PAD]':0, 'O':1, }
    for item in data:

        converted = {}
        converted['sentence'] = " ".join(item["token"])
        converted['relation'] = item['relation']
        if converted['relation'] not in label2id:
            label2id[converted['relation']] = len(label2id)

        converted["stanford_ner"] = item["stanford_ner"]
        for ner in converted["stanford_ner"]:
            if ner not in ner2id:
                ner2id[ner] = len(ner2id)

        h_start = item['subj_start']
        h_end = item['subj_end']
        head = " ".join(item["token"][h_start:h_end+1])
        # h_char_end = len(" ".join(item["token"][:h_end+1]))
        # h_char_start = h_char_end - len(head)
        converted['head'] = {'word': head, 'pos':[h_start, h_end], 'type':item["subj_type"]}

        t_start = item['obj_start']
        t_end = item['obj_end']
        tail = " ".join(item["token"][t_start:t_end+1])
        # t_char_end = len(" ".join(item["token"][:t_end+1]))
        # t_char_start = t_char_end - len(tail)
        converted['tail'] = {'word':tail, 'pos':[t_start, t_end], 'type':item["obj_type"]}

        converted_data.append(converted)
        pair = pairdict.get((head, tail), {})
        pair[converted['relation']] = pair.get(converted['relation'], 0) + 1
        pairdict[(head, tail)] = pair
        labeldict[label2id[converted['relation']]] += 1

    labelnum = [labeldict[i] for i in range(len(label2id))]

    id2label = {v:k for k,v in label2id.items()}
    filtered_instances = []
    count_noise = 0
    label_noise = {k: 0 for k in label2id.keys()}
    label_num = {k: 0 for k in label2id.keys()}
    for ins in converted_data:
        if mode == "train":
            head = ins["head"]["word"]
            tail = ins['tail']['word']
            relation = ins["relation"]
            pair = pairdict[(head, tail)]

            if random.uniform(0, 1) > 0.7:
                count_noise += 1
                while True:
                    weight = [float(i) / sum(labelnum) for i in labelnum]
                    neg = random.choices([i for i in range(0, len(label2id))]
                                         , k=1, weights=weight)[0]
                    if neg != label2id[relation]:
                        ins["D_relation"] = id2label[neg]
                        label_num[ins["D_relation"]] += 1
                        label_noise[ins["D_relation"]] += 1
                        break
            else:
                ins["D_relation"] = ins["relation"]
                label_num[ins["D_relation"]] += 1
                # filtered_instances.append(ins)

        else:
            ins["D_relation"] = ins["relation"]
            label_num[ins["D_relation"]] += 1
            # filtered_instances.append(ins)

    for k in label_num.keys():
        print('Label {} num: {}, noise num: {}'.format(k, label_num[k], label_noise[k]))
    print('Filtered data has {} instances, noise num: {}'.format(len(converted_data), count_noise))
    print(len(converted_data)-label_num["no_relation"])
    with open(output_file, 'w') as outfile:
       json.dump(converted_data, outfile)

    # with open('/home/mrt/data/tacred/data/ner2id.json', 'w') as outfile:
    #     json.dump(ner2id, outfile)
