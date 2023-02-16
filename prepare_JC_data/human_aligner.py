"""Human-power aligner."""

from __future__ import annotations

import argparse

def main():
    parser = argparse.ArgumentParser('Human-power aligner.')
    parser.add_argument('terms1', help='Term list 1')
    parser.add_argument('terms2', help='Term list 2')
    parser.add_argument('txt1', help='Text 1')
    parser.add_argument('txt2', help='Text 1')
    parser.add_argument('result', help='Output alignment file.')
    args = parser.parse_args()

    terms1: list[list[str]] = [l.split() for l in open(args.terms1)]
    terms2: list[list[str]] = [l.split() for l in open(args.terms2)]
    txt1: list[str] = [l.strip() for l in open(args.txt1)]
    txt2: list[str] = [l.strip() for l in open(args.txt2)]
    n: int = len(terms1)
    assert len(terms2) == n
    assert len(txt1) == n
    assert len(txt2) == n 
    result: list[list[tuple[int, int]]] = [[] for _ in range(n)]

    cur = 0
    while True:
        try:
            print()
            print('----------------')
            print(f'Line {cur + 1}')
            print(f'txt1: {txt1[cur]}')
            print(f'txt2: {txt2[cur]}')
            print()
            
            print(f'Alignments:')
            for i, (t1, t2) in enumerate(result[cur]):
                print(f'{i: 4d} {terms1[cur][t1]} --- {terms2[cur][t2]}')
            print()
            
            print(f'Terms 1:')
            for i, t1 in enumerate(terms1[cur]):
                print(f'{i: 4d} {t1}')
            print()
            
            print(f'Terms 2:')
            for i, t2 in enumerate(terms2[cur]):
                print(f'{i: 4d} {t2}')
            print()

            command = input('>>> ')
            command = command.split()
            if command[0] == 'exit':
                break
            elif command[0] == 'n':
                cur = min(cur + 1, n - 1)
            elif command[0] == 'p':
                cur = max(cur - 1, 0)
            elif command[0] == 'g':
                cur = max(min(int(command[1]), n - 1), 0) + 1
            elif command[0] == 'a':
                t1 = int(command[1])
                t2 = int(command[2])
                assert 0 <= t1 < len(terms1[cur])
                assert 0 <= t2 < len(terms2[cur])
                result[cur].append((t1, t2))
            elif command[0] == 'd':
                r = int(command[1])
                assert 0 <= r < len(result[cur])
                del result[cur][r]
        except Exception as ex:
            # print(type(ex).__name__)
            pass
    
    with open(args.result, 'w') as fp:
        for i, r in enumerate(result):
            for t1, t2 in r:
                print(f'{i + 1}\t{terms1[i][t1]}\t{terms2[i][t2]}', file=fp)


if __name__ == '__main__':
    main()