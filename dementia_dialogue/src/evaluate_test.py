from sklearn.metrics import precision_score, recall_score, f1_score
import logging 
import argparse

# logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s : %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %p %I:%M:%S')
logging.info("evaluate pred_data ... ")

# args
parser = argparse.ArgumentParser("evaluate predicted data with gold data")
parser.add_argument('-pred', type=str, help="Path to predicted data")
parser.add_argument('-gold', type=str, help="Path to gold data")
parser.add_argument('--average', type=str, default = "macro", help="how to average when calculating each score (default : macro)")

args = parser.parse_args()

pred_path = args.pred
gold_path = args.gold
average = args.average

with open(pred_path) as f_pred, open(gold_path) as f_gold:
    f_pred.readline()
    pred_labels = [pred_line.strip().split('\t')[1] for pred_line in f_pred]
    gold_labels = [gold_line.strip().split('\t')[1] for gold_line in f_gold]

    assert len(pred_labels) == len(gold_labels), "not agreement between pred_data and gold_data"

    precision = precision_score(gold_labels, pred_labels, average=average)
    recall = recall_score(gold_labels, pred_labels, average=average)
    f1_score = f1_score(gold_labels, pred_labels, average=average)

    print("precision : {}".format(precision))
    print("recall : {}".format(recall))
    print("f1_score : {}".format(f1_score))

    
