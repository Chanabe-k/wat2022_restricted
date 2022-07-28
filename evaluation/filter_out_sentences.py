# Usage: 
#  (en) python evaluation/filter_out_sentences.py --dic ../data/dic/test.dic.en --input ../data/sample_output.en --output ../work/filtered_sample_output.en --lang en
#  (ja) python evaluation/filter_out_sentences.py --dic ../data/dic/test.dic.ja --input ../data/sample_output.ja --output ../work/filtered_sample_output.ja --lang ja

from tqdm import tqdm
import re
import argparse

def _lowercase(text: str):
    return text.lower()

# def _detokenize(text: str):
#     return ''.join(text.split(' '))

# def _remove_punctuation(text: str):
#     punctuations = r'\'"「」\()（）'
#     for punc in punctuations:
#         text = text.replace(punc, '')
#     return text

def _replace_specialchar(w: str):
    specialchars = '+()'
    for specialchar in specialchars:
        w = w.replace(specialchar, '\\' + specialchar)
    return w

def filter_sentence_with_dic(s: str, dic: list, lang: str):
    orig_s = s
    
    # Check whether the sent contains all the restricted words
    match_count = 0

    for w in dic:
        if lang == 'en':
            for f in [_lowercase]: # , _detokenize, _remove_punctuation
                w, s = f(w), f(s)
    
        w = _replace_specialchar(w)
        
        if re.search(w, s):
            match_count += 1
    
    # Filter
    if match_count == len(dic):
        return orig_s
    else:
        return ''

def get_dic(file_path: str):
    """Load dictionary information (get dic_list)
    input: file_path(str)"""
    c = []
    for l in open(file_path):
        l = l.strip()
        if l:
            c.append(l)
        else:
            yield c
            c = []        
    yield c
        
def main():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--dic', type=str, help='Path to dictionary file')
    parser.add_argument('--input', type=str, help='Path to Input file (model output)')
    parser.add_argument('--output', type=str, help='Filterd output name')
    parser.add_argument('--lang', type=str, choices = ['en', 'ja'], help='Input Language')
    args = parser.parse_args()

    dic_path = args.dic
    input_path = args.input
    filtered_output_path = args.output

    dic_list = list(get_dic(dic_path))
    print(f"dic length: {len(dic_list[:-1])}")

    with open(filtered_output_path, 'w') as f_filtered:
        # Exact match between dictionary and output 
        with open(input_path) as f_out:
            # Check file length == dictionary length
            sents = [l.strip() for l in f_out]
            assert len(sents) == len(dic_list[:-1]), "not match file length and dictionary length"
            
            # Filter out sentences not containing restricted vocab
            for sent, dic in tqdm(zip(sents, dic_list[:-1])):
                filtered_sent = filter_sentence_with_dic(sent, dic, args.lang)
                f_filtered.write(filtered_sent + '\n')
        print(f"> {filtered_output_path}")

main()