import codecs
from tqdm import tqdm
import logging 
import argparse

# logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s : %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %p %I:%M:%S')
logging.info("write extracted_data ... ")

# args
parser = argparse.ArgumentParser("extract important data from original data")
parser.add_argument('output', help="output file name")
parser.add_argument('-t', '--time', action="store_true", help="add time annotation")
parser.add_argument('-i', '--intention', action="store_true", help = 'add intention annotation')
args = parser.parse_args()

# input, output
csv_path = "./data/annotationsのコピー2.csv"
# csvって言ってるけどtsvだった
extracted_tsv_path = args.output

# main
logging.info("filename: '{}'".format(extracted_tsv_path))
with codecs.open(csv_path, encoding="utf-16") as f_csv, open(extracted_tsv_path, 'w') as f_w:
    header = f_csv.readline().strip()
    print(header)
    header = header.split('\t')
    added_annotations = []
    if args.intention:
        added_annotations.append("intention")
    if args.time:
        added_annotations.append("time")
    header_list = [h for h in header if h.strip() in ["id", "file_id", "script"] + added_annotations]
    w_header = '\t'.join(header_list) + '\n'
    f_w.write(w_header)
    j = 0
    # 0: id, 1: file_id, 6: speaker, 8: script, 13: intention, 22: time
    for i, line in tqdm(enumerate(f_csv)):
        line = line.strip()
        split_line = line.split('\t')
        sent_id, file_id, script, intention, time = split_line[0], split_line[1], split_line[8], split_line[13], split_line[22]
        
        columns = [sent_id, file_id, script]
        if args.intention:
            columns.append(intention)
        if args.time:
            columns.append(time)
        
        if len(time.strip()) > 1:
            j += 1
            writeline = '\t'.join(columns) + '\n'
            f_w.write(writeline)
    #print(w_header)
    #print(writeline)
    logging.info("whole sentences: " + "{}(/{})".format(str(j), str(i)))
        