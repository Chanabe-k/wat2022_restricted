# timelabelだけ抽出 & timelabelをid化したcolumn作成
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

# label_set
timelabel_set = ['過去', '過去-最近', '最近（1か月以内）', '現在（状態、性質、考えなど）', '過去-現在（習慣など）', '未来（予定、予測、願望、仮定など）', '現在-未来', '最近-未来', '過去-未来', '最近-現在（習慣など）']
timelabel2id = {label : str(i) for i, label in enumerate(timelabel_set)}
logging.info(timelabel2id)

with codecs.open(csv_path, encoding="utf-16") as f_csv, open(extracted_tsv_path, 'w') as f_w:
    header = f_csv.readline().strip()
    
    header = header.split('\t')
    # 抽出したいannotaitonをリスト化
    new_header = ["id", "file_id", "script"]

    if args.intention:
        new_header.append("intention")
    if args.time:
        new_header.append("time")
        new_header.append("time_id")
    
    w_header = '\t'.join(new_header) + '\n'
    print(w_header)
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
            if len(time.strip()) > 1:
                time_id = timelabel2id[time]
                if time_id not in ['1', '6', '7', '8', '9']:
                    columns.append(time)
                    columns.append(timelabel2id[time])
                    writeline = '\t'.join(columns) + '\n'
                    f_w.write(writeline)
                    j += 1
    
    logging.info("whole sentences: " + "{}(/{})".format(str(j), str(i)))
        