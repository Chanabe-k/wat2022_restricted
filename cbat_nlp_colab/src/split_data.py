import logging
from itertools import groupby
import json

# logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s : %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %p %I:%M:%S')
 
tmp_path = "./tmp/scripts_time.tsv.tok"
train_path = "./data/train.tok"
valid_path = "./data/valid.tok"
test_path = "./data/test.tok"

# divide into train/dev/test set
logging.info("read file: {}".format(tmp_path))
with open(tmp_path) as f_tmp:
    f_tmp.readline()

    data_dic = {}
    all_data = [(file_id, list(lines)) for file_id, lines in groupby(f_tmp, lambda line: line.strip().split()[1])]
    train = all_data[:150]
    valid = all_data[150:187]
    test = all_data[187:]
    logging.info("dataset divided: {}".format(str(list(map(len, [train, valid, test])))))

# write train/valid/test file in data directory
logging.info("write file: {}".format(' '.join([train_path, valid_path, test_path])))

def write_data(data_fobj, data_list):
    for _, lines in data_list:
        for line in lines:
            line = line.strip()
            #_, _, sent, _, label, _  = line.split('\t')
            #data_fobj.write(sent + '\t' + label + '\n')
            _, _, _, _, label, tokenized_sent  = line.split('\t')
            data_fobj.write(tokenized_sent + '\t' + label + '\n')

with open(train_path, 'w') as f_train, open(valid_path, 'w') as f_valid, open(test_path, 'w') as f_test:
    write_data(f_train, train)
    write_data(f_valid, valid)
    write_data(f_test, test)

