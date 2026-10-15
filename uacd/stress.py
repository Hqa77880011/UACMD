import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--missing-rate', type=float, default=.3)
    parser.add_argument('--pattern', choices=['random', 'block', 'sensor'], default='random')
    parser.add_argument('--sensor', type=int, default=0)
    parser.add_argument('--block', type=int, default=20)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    if not 0 <= args.missing_rate <= 1:
        parser.error('missing-rate must lie in [0,1]')
    data = dict(np.load(args.input, allow_pickle=False))
    x = data['x'].copy()
    n, m = x.shape[:2]
    available = data.get('available', np.ones((n, m), dtype=bool)).astype(bool)
    selected = np.flatnonzero(data['split'] == 'test')
    rng = np.random.default_rng(args.seed)
    if args.pattern == 'random':
        available[selected] &= rng.random((len(selected), m)) >= args.missing_rate
    elif args.pattern == 'sensor':
        available[selected, args.sensor] = False
    else:
        groups = data['group'][selected]
        for group in np.unique(groups):
            ids = selected[groups == group]
            for start in range(0, len(ids), args.block):
                if rng.random() < args.missing_rate:
                    available[ids[start:start + args.block], args.sensor] = False
    empty = selected[~available[selected].any(1)]
    for row in empty:
        candidates = np.flatnonzero(data.get('available', np.ones((n,m), dtype=bool))[row])
        if not len(candidates):
            raise ValueError('An original sample has no available modality')
        available[row, rng.choice(candidates)] = True
    if args.noise:
        x[selected, args.sensor] += rng.normal(0, args.noise, x[selected, args.sensor].shape)
    data['x'], data['available'] = x, available
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)


if __name__ == '__main__':
    main()
