#!/bin/python

lst = ["dev", "devtest", "test"]
langs = ["ja", "zh"]

for val in lst:
    for l in langs:
        print("(val, l): {}, {}".format(val, l))
        with open(val + ".dic." + l) as f:
            sen_count = 1
            phrase_count = 0
            char_count = 0
            for line in f:
                line = line.strip()
                if len(line) > 0:  # phrase exists
                    phrase_count += 1
                    char_count += len(line)
                else:
                    sen_count += 1.
            avg_phrase = phrase_count / sen_count
            avg_char = char_count / sen_count
            print("avg_phrase: ", avg_phrase)
            print("avg_chars: ", avg_char)
            print("line: ", sen_count)
            
