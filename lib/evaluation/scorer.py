#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import numpy as np
from sklearn.metrics import auc
import json
# NO_RELATION = ""

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def score(key, prediction, verbose=False, NO_RELATION='NA'):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    # print( "Precision (micro): {:.3%}".format(prec_micro) )
    # print( "   Recall (micro): {:.3%}".format(recall_micro) )
    # print( "       F1 (micro): {:.3%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro


def curve(y_scores, y_true, num=2000):
    order = np.argsort(y_scores)[::-1]
    guess = 0.
    right = 0.
    target = np.sum(y_true)
    precisions = []
    recalls = []
    for o in order[:num]:
        guess += 1
        if y_true[o] == 1:
            right += 1

        precision = right / guess
        recall = right / target
        precisions.append(precision)
        recalls.append(recall)
    return np.array(recalls), np.array(precisions)


def AUC_and_PN(y_scores, y_true):

    recalls, precisions = curve(y_scores, y_true, 3000)
    recalls_01 = recalls[recalls < 0.1]
    precisions_01 = precisions[recalls < 0.1]
    AUC_01 = auc(recalls_01, precisions_01)

    recalls_02 = recalls[recalls < 0.2]
    precisions_02 = precisions[recalls < 0.2]
    AUC_02 = auc(recalls_02, precisions_02)

    recalls_03 = recalls[recalls < 0.3]
    precisions_03 = precisions[recalls < 0.3]
    AUC_03 = auc(recalls_03, precisions_03)

    recalls_04 = recalls[recalls < 0.4]
    precisions_04 = precisions[recalls < 0.4]
    AUC_04 = auc(recalls_04, precisions_04)

    AUC_all = average_precision_score(y_true, y_scores)

    print(AUC_01, AUC_02, AUC_03, AUC_04, AUC_all)

    for q, testdata in enumerate([test1, test2, testall]):

        y_true, y_scores = eval(model, testdata, args)

        order = np.argsort(-y_scores)

        top100 = order[:100]
        correct_num_100 = 0.0
        for i in top100:
            if y_true[i] == 1:
                correct_num_100 += 1.0
        print('P@100: ', correct_num_100 / 100)

        top200 = order[:200]
        correct_num_200 = 0.0
        for i in top200:
            if y_true[i] == 1:
                correct_num_200 += 1.0
        print('P@200: ', correct_num_200 / 200)

        top300 = order[:300]
        correct_num_300 = 0.0
        for i in top300:
            if y_true[i] == 1:
                correct_num_300 += 1.0
        print('P@300: ', correct_num_300 / 300)

        print('mean: ', (correct_num_100 / 100 + correct_num_200 / 200 + correct_num_300 / 300) / 3)



def bag_eval(pred_result, facts):
    """
    Args:
        pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
            Note that relation of NA should be excluded.
    Return:
        {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
            prec (precision) and rec (recall) are in micro style.
            prec (precision) and rec (recall) are sorted in the decreasing order of the score.
            f1 is the max f1 score of those precison-recall points
    """
    sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec = []
    rec = []
    correct = 0
    preds = []
    filtered_facts = {}
    total = sum([len(l) for l in facts.values()])
    count = 0
    for i, item in enumerate(sorted_pred_result):
        if  len(facts[item['entpair']]) == 0:
            continue
        if correct != total :
            preds.append((item['entpair'], "pred: {}, gold: {}".format(item['relation'], str(facts[item['entpair']]))))

        if item['relation'] in facts[item['entpair']]:
            correct += 1
        count += 1
        prec.append(float(correct) / float(count))
        rec.append(float(correct) / float(total))

    for c in [100, 200, 300]:
        print("P@{}: {}".format(c, prec[c]))

    auc_result = auc(x=rec, y=prec)
    np_prec = np.array(prec)
    np_rec = np.array(rec)
    f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    mean_prec = np_prec.mean()

    with open("bagtest_result5.txt", 'w') as f:
        json.dump(prec, f)
        json.dump(rec, f)
        f.write("\n")
        for line in preds:
            f.write(str(line)+"\n")
    #     json.dump(np_prec,f)
    #     json.dump(np_rec, f)

    return {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'auc': auc_result}



if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

