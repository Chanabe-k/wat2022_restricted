"""split original data (ASPEC-JC) to *.{ja/zh or en}"""

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-input', type=str, help='Path to input file')
    parser.add_argument('-target_lang', type=str, help='target language')
    parser.add_argument('--output_dir', type=str, help='Path to output directory', default='.')
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir
    ja_output = output_dir + '/' + os.path.basename(input_file) + '.ja'
    target_output = output_dir + '/' + os.path.basename(input_file) + f'.{args.target_lang}'

    all_data = [line.split(' ||| ') for line in open(input_file)]

    if args.target_lang == 'zh':
        ja_data = [data[1] for data in all_data]
        target_data = [data[2] for data in all_data]
    elif args.target_lang == 'en':
        ja_data = [data[2] for data in all_data]
        target_data = [data[3] for data in all_data]

    with open(ja_output, 'w') as f_ja, open(target_output, 'w') as f_t:
        for j_data, t_data in zip(ja_data, target_data):
            f_ja.write(j_data.strip() + '\n')
            f_t.write(t_data.strip() + '\n')
        print(f'Write {ja_output}')
        print(f'Write {target_output}')