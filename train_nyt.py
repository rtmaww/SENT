import sys
import os
import argparse
sys.path.append(os.path.abspath('lib/'))
from dataloader.Data_nyt import Data
from model.sent_model import SENT_Model
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from tqdm import tqdm, trange
import pickle
from transformers.optimization import *
from evaluation import scorer
from collections import defaultdict
import copy

def load_data(args, mode='train'):

    data_path = args.save_data_path + '.' + mode + '.data'
    if os.path.exists(data_path):
        print('Loading {} data from {}...'.format(mode, data_path))
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = Data(args, mode)
        print('Saving {} data to {}...'.format(mode, data_path))
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    return data


def train_SN(args):

    data_train, data_dev, data_test = load_data(args, 'train'), load_data(args, 'dev'), load_data(args, 'test')

    if args.noise_label:
        data_noise = load_data(args, 'test_noise')
    else:
        data_noise = None

    args.label_size = len(data_test.relId2labelId)
    args.ner_label_size = len(data_test.ner2id)

    trainer = Trainer(args, data_train, data_dev, data_test, data_noise)

    if args.mode == 'train':
        trainer.train_batch()
        trainer.test_batch()

    elif args.mode == 'test':
        try:
            trainer.load_model(args.load_model_name)
            print('Loading model from {}'.format(args.load_model_name))
        except:
            trainer.load_model(args.save_model_name)
            print('Loading model from {}'.format(args.save_model_name))
        eval_result = trainer.test_batch(data='dev')
        eval_result = trainer.test_batch()


class Trainer(nn.Module):

    def __init__(self, options, data_train, data_dev, data_test, data_noise):
        super(Trainer, self).__init__()
        self.options = options
        self.batch_size = options.batch_size
        self.save_path = options.save_model_name
        self.device = options.gpu
        self.n_device = options.n_gpu
        self.test_noise = options.noise_label
        self.n_initial_epoch = 15
        self.n_iter_epoch = 10
        self.n_iter_num = 15
        self.n_posi_epoch = 15
        self.options.random = False

        if data_train is not None:
            self.data_train = data_train
            self.id2label = data_train.labelId2rel
            train_batch_data = data_train.batchify()
            self.ori_train_labels = [d[-1].item() for d in train_batch_data]
            self.train_dl = DataLoader(train_batch_data, sampler=RandomSampler(train_batch_data),
                                       batch_size=args.batch_size)

        if self.test_noise:
            self.data_noise = data_noise
            noise_batch_data = data_noise.batchify(noise_label=True)
            for idx in range(len(noise_batch_data)):
                if noise_batch_data[idx][-2] == 0 and noise_batch_data[idx][-1]:
                    noise_batch_data[idx][-1] = False
            self.test_noise_dl = DataLoader(noise_batch_data, sampler=SequentialSampler(noise_batch_data),
                                       batch_size=args.batch_size)
            self.pred_noise = [0] * len(noise_batch_data)

        self.data_dev = data_dev
        self.data_test = data_test

        self.eval_interval = 10

        test_batch_data = data_test.batchify()
        dev_batch_data = data_dev.batchify()

        self.test_dl = DataLoader(test_batch_data, sampler=SequentialSampler(test_batch_data), batch_size=args.batch_size)
        self.dev_dl = DataLoader(dev_batch_data, sampler=SequentialSampler(dev_batch_data), batch_size=args.batch_size)

        self.initialize()


    def initialize(self):
        print("Initializing model...")

        self.options.vocab_size = self.data_test.get_vocab_size()
        self.SENTmodel = SENT_Model(self.options, vocab_file=self.options.word2vec_file)
        if self.n_device > 1:
            self.SENTmodel = torch.nn.DataParallel(self.SENTmodel)
        if self.device:
            self.SENTmodel = self.SENTmodel.to(self.device)

        ## ================= Setup Optimizer ========================
        weight_decay = self.options.weight_decay

        self.para_list = [p for p in self.SENTmodel.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.para_list, lr=self.options.lr, weight_decay=weight_decay)


    def train_batch(self):

        # print("-----training phrase-----")

        self.options.random = False

        # Initial training
        print('------------ Start Initial Training -------------')
        self.save_path = self.save_path + "-N0"
        self.train_epoch(self.n_initial_epoch, negloss=True, metric="train")


        # Iterative training
        print('------------- Start Iterative Training -------------')

        for iter in range(self.n_iter_num):
            print('Iterative Training Phase {}'.format(iter+1))
            self.filter_relabel(prob_threshold=0.25, cutrate=0.01, relabel_rate=0.7)
            self.test_denoise(prob_threshold=0.25, cutrate=0.01)
            self.initialize() # re-initialize
            self.save_path = "-".join(self.save_path.split("-")[:-1]) + f"-N{iter+1}"
            self.train_epoch(self.n_iter_epoch, negloss=True, metric="dev")


        # Finish iterative training, start positive training
        print('------------- Start Positive Training -------------')

        self.filter_relabel(0.25, cutrate=0., relabel_rate=0.7)
        self.test_denoise(0.25, cutrate=0.)
        self.options.random = True   # set random because the baseline method is randomly initialized.
        self.initialize()
        print('Start Positive Training')
        self.save_path = "-".join(self.save_path.split("-")[:-1]) + '-P'
        self.train_epoch(self.n_posi_epoch, negloss=False, metric="dev", test=True)



    def train_epoch(self, epoch_num, negloss=False, test=False, metric="dev"):

        best_metric = -1
        for epoch in range(epoch_num):
            total_loss = 0.
            all_right = 0.
            all_total = 0.
            all_pos_right = 0.
            all_pos_total = 0.
            idx = 0
            predictions = []
            true_labels = []
            print('------epoch {}/{}------'.format(epoch+1, epoch_num))
            for i, train_batch in tqdm(enumerate(self.train_dl)):
                self.SENTmodel.train()
                idx += train_batch[0].size(0)
                train_batch = [i.to(self.device) for i in train_batch]
                loss, preds, right, total, pos_right, probs, _ = self.SENTmodel(train_batch, negloss=negloss)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                all_right += right.sum().item()
                all_total += total.sum().item()
                all_pos_right += pos_right.sum().item()
                all_pos_total += (train_batch[-1] != 0).sum().item()
                predictions += preds.cpu().squeeze().tolist()
                true_labels += train_batch[-1].cpu().squeeze().tolist()

            acc = all_right / all_total
            pos_acc = all_pos_right / all_pos_total

            print('Epoch {} finished, total loss={}, distant acc={}, pos acc={} '.format(epoch, total_loss, acc, pos_acc))

            predictions_ = []
            true_labels_ = []

            for pred, true in zip(predictions, true_labels):
                predictions_.append(self.data_test.id2rel(pred))
                true_labels_.append(self.data_test.id2rel(true))

            train_p, train_r, train_f1 = scorer.score(true_labels_, predictions_, False, 'NA')
            print('Training set P={}, R={}, F1={} on real labels'.format(train_p, train_r, train_f1))

            dev_acc, dev_f1 = self.test_batch('dev')

            if test:
                test_acc, test_f1 = self.test_batch('test')

            if metric == 'train':
                if train_f1 > best_metric:
                    best_metric = train_f1
                    torch.save(self.SENTmodel.state_dict(), self.save_path)
                    print('Saving model to {}...'.format(self.save_path))
            else:
                if dev_f1 > best_metric:
                    best_metric = dev_f1
                    torch.save(self.SENTmodel.state_dict(), self.save_path)
                    print('Saving model to {}...'.format(self.save_path))


    def filter_relabel(self, prob_threshold=0., cutrate=-1, relabel_rate=-1, convergence_rate=0.99):

        print('Filtering with label rank...')

        print("Loading model from {}".format(self.save_path))
        self.SENTmodel.load_state_dict(torch.load(self.save_path))
        self.SENTmodel.eval()

        train_batch_data = self.train_dl.dataset
        for i in range(len(train_batch_data)):
            train_batch_data[i][-1][:] = self.ori_train_labels[i]
        self.train_dl = DataLoader(train_batch_data, batch_size=self.batch_size, sampler=SequentialSampler(train_batch_data))

        all_probs = []
        max_probs = []
        all_preds = []
        for i, train_batch in tqdm(enumerate(self.train_dl)):
            train_batch = [i.cuda() for i in train_batch]
            with torch.no_grad():
                loss, preds, right, total, r_right, probs, label_probs = self.SENTmodel(train_batch, mode='test')

            all_probs += probs.view(-1).cpu().numpy().tolist()
            max_probs += label_probs.max(-1)[0].view(-1).cpu().numpy().tolist()
            all_preds += preds.view(-1).cpu().numpy().tolist()

        filtered_dataset = self.train_dl.dataset
        noisy_data_index = []
        filtered_index = []
        prob_dict = defaultdict(list)
        for index, (data, prob) in enumerate(zip(self.train_dl.dataset, all_probs)):
            prob_dict[data[-1].item()].append([index, prob])

        relabel_cnt = defaultdict(int)
        prob_dict = {label:sorted(probs, key=lambda x:x[1]) for label, probs in prob_dict.items()}
        for label, sorted_probs in prob_dict.items():
            th = 0
            prob = 0
            if cutrate>0 and sorted_probs[-1][1] < cutrate:
                pass
            else:
                prob = 2 * prob_threshold * sorted_probs[-1][1] if sorted_probs[-1][1] > convergence_rate \
                    else prob_threshold * sorted_probs[-1][1]

                for i, (index, p) in enumerate(sorted_probs):
                    if p < prob:
                        noisy_data_index.append(index)
                        th += 1

                        # if relabel_rate is set, relabel if the highest prob value > relabel_rate
                        if relabel_rate > 0:
                            if max_probs[index] > relabel_rate:
                                filtered_dataset[index][-1][:] = all_preds[index]  # re-label
                                relabel_cnt[all_preds[index]] += 1
                            else:
                                filtered_dataset[index][-1][:] = 0  # set label to NA
                                relabel_cnt[0] += 1
                        else:
                            filtered_dataset[index][-1][:] = 0  # set label to NA
                    else:
                        filtered_index.append(index)


            print("Filtering {}/{} instance with label {}, threshold prob={}, max prob={}"
                  .format(th, len(prob_dict[label]), self.id2label[label], prob, sorted_probs[-1][1]))

        print("-----------Relabel detail------------")
        for key, value in relabel_cnt.items():
            print("Relabel {} for label {}, Th={}".format(value, self.id2label[key], prob_dict[key][-1][1] ))

        # #### calculating denoise statics
        pred_noise_num = float(len(noisy_data_index))

        print('Deleting {} noisy instances with threshold={}'.format(pred_noise_num,
                                                                                     prob_threshold))

        # re-set the train dataloader with re-fined training data
        self.train_dl = DataLoader(filtered_dataset, batch_size=self.batch_size, sampler=RandomSampler(filtered_dataset))



    def test_denoise(self, prob_threshold=0., cutrate=-1, convergence_rate=0.99, retain=True):

        print('Test denoise ability')

        print("Loading model from {}".format(self.save_path))
        self.SENTmodel.load_state_dict(torch.load(self.save_path))

        self.SENTmodel.eval()

        if retain:
            ori_noise = copy.deepcopy(self.pred_noise)

        all_probs = []
        max_probs = []
        all_preds = []
        for i, train_batch in tqdm(enumerate(self.test_noise_dl)):
            train_batch, is_noise = train_batch[:-1], train_batch[-1]
            train_batch = [i.cuda() for i in train_batch]
            with torch.no_grad():
                loss, preds, right, total, r_right, probs, label_probs = self.SENTmodel(train_batch, mode='test')

            all_probs += probs.view(-1).cpu().numpy().tolist()
            max_probs += label_probs.max(-1)[0].view(-1).cpu().numpy().tolist()
            all_preds += preds.view(-1).cpu().numpy().tolist()

        noisy_data_index = []
        filtered_index = []
        prob_dict = defaultdict(list)
        for index, (data, prob) in enumerate(zip(self.test_noise_dl.dataset, all_probs)):
            prob_dict[data[-2].item()].append([index, prob])


        prob_dict = {label: sorted(probs, key=lambda x: x[1]) for label, probs in prob_dict.items()}
        for label, sorted_probs in prob_dict.items():
            th = 0
            right = 0.
            gold = 0.
            pred = 0.
            prob = 0.
            if cutrate>0 and sorted_probs[-1][1] < cutrate:
                for index, p in sorted_probs:
                    gold += self.test_noise_dl.dataset[index][-1]

            else:
                prob = 2 * prob_threshold * sorted_probs[-1][1] if sorted_probs[-1][1] > convergence_rate \
                    else prob_threshold * sorted_probs[-1][1]
                for i, (index, p) in enumerate(sorted_probs):
                    gold += self.test_noise_dl.dataset[index][-1]
                    if p < prob:
                        noisy_data_index.append(index)
                        th += 1

                        right += self.test_noise_dl.dataset[index][-1]
                        pred += 1
                        if self.test_noise_dl.dataset[index][-2] != 0:
                            self.pred_noise[index] = 1
                    else:
                        filtered_index.append(index)

            if label == 0 :
                right = 0.
                pred = 0.
                gold = 0.

            if pred == 0:
                p = 0.
            else:
                p = float(right)/ float(pred)
            if gold == 0:
                r = 0.
            else:
                r = float(right) / float(gold)

            print("Filtering {}/{} instance with label {}, threshold prob={}, max prob={}, p={}, r={}"
                  .format(th, len(prob_dict[label]), self.id2label[label], prob, sorted_probs[-1][1], p, r))

        # #### calculating global denoise statistics
        real_noise_num = 0.
        right_noise_num = 0.
        pred_noise_num = 0.
        for idx in range(len(self.test_noise_dl.dataset)):
            if self.test_noise_dl.dataset[idx][-1]:
                real_noise_num += 1
                if self.pred_noise[idx] == 1:
                    right_noise_num += 1
            if self.pred_noise[idx] == 1:
                pred_noise_num += 1
        if pred_noise_num == 0:
            noise_p = 0.
        else:
            noise_p = right_noise_num / pred_noise_num
        if real_noise_num == 0:
            noise_r = 0.
        else:
            noise_r = right_noise_num / real_noise_num

        print('Deleting {} noisy instances with threshold={}, p={}, r={}'.format(pred_noise_num,
                                                                                     prob_threshold, noise_p, noise_r))
        if retain:
            self.pred_noise = ori_noise


    def test_batch(self, data='test', detail=False):
        if data == 'test':
            print("-----testing phrase-----")
            dl = self.test_dl
        elif data == 'dev':
            print("------dev phrase------")
            dl = self.dev_dl

        predictions = []
        true_labels = []
        # total_loss = 0.
        all_right = 0.
        all_total = 0.
        all_pos_right = 0.
        all_pos_total = 0.
        all_probs = []
        for i, test_batch in tqdm(enumerate(dl)):
            self.SENTmodel.eval()
            test_batch = [i.cuda() for i in test_batch]
            with torch.no_grad():
                loss, preds, right, total, pos_right, probs, _ = self.SENTmodel(test_batch, mode='test')
            predictions += preds.cpu().squeeze().tolist()
            true_labels += test_batch[-1].cpu().squeeze().tolist()
            all_right += right.sum().item()
            all_total += total.sum().item()
            all_pos_right += pos_right.sum().item()
            all_pos_total += (test_batch[-1] != 0).sum().item()
            all_probs += probs.cpu().numpy().tolist()
        acc = all_right / all_total
        pos_acc = all_pos_right / all_pos_total
        print('Finish evaluating on {} set, distant acc={}, pos acc={} '.format(data, acc, pos_acc))
        predictions = [self.data_test.id2rel(i) for i in predictions]
        true_labels = [self.data_test.id2rel(i) for i in true_labels]
        test_p, test_r, test_f1 = scorer.score(true_labels, predictions, detail, 'NA')
        print('P={}, R={}, F1={}'.format(test_p, test_r, test_f1))
        return pos_acc, test_f1


    def load_model(self, load_path):
        if os.path.exists(load_path):
            self.SENTmodel.load_state_dict(torch.load(load_path))
            print('Loading model from {}...'.format(load_path))
        elif os.path.exists(self.save_path):
            self.SENTmodel.load_state_dict(torch.load(self.save_path))
            print('Loading model from {}...'.format(self.save_path))
        else:
            print('Fail to load model from {}'.format(load_path))
            return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--dataset",type=str,default='ori')
    parser.add_argument("--train_data_file",type=str,default='data/train_ner.json')
    parser.add_argument("--dev_data_file",type=str,default='data/dev_part_ner.json')
    parser.add_argument("--test_data_file",type=str,default='data/test_part_ner.json')
    parser.add_argument("--rel2id_file", type=str, default='data/rel2id.json')
    parser.add_argument("--vocab_file", type=str,
                        default='data/glove/glove.6B.50d_word2id.json')
    parser.add_argument("--word2vec_file", type=str,
                        default='data/glove/glove.6B.50d_mat.npy')
    parser.add_argument("--load_model_name",type=str,default=None)
    parser.add_argument("--save_model_path",type=str,default='savemodel/')
    parser.add_argument("--save_model_name",type=str,default='model')
    parser.add_argument("--save_data_path",type=str,default='data/')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr",type=float,  default = 5e-4)
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--noise_label', action='store_true',
                        help='Test noise label')
    args = parser.parse_args()

    if args.noise_label:
        args.test_noise_file = "data/test_noise_ner.json"

    args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    args.save_model_name = args.save_model_path + args.save_model_name
    train_SN(args)
