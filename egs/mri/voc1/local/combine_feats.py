#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Combine two sets of features.

E.g., `python3 local/combine_feats.py downloads/emadata/cin_us_mngu0 --feats pitch actions -o fnema`
    concatenates F0 sequences (1-dim.) with EMA sequences (12-dim.), yielding 13-dim. sequences
"""

import argparse
import numpy as np
import os

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('d')
parser.add_argument('--feats', nargs='+', required=True)
parser.add_argument('-o', required=True)
args = parser.parse_args()

oud = os.path.join(args.d, args.o)
if not os.path.exists(oud):
    os.makedirs(oud)

d = os.path.join(args.d, args.feats[0])
fs = os.listdir(d)
fs = [f for f in fs if f.endswith('.npy')]
fs_set = set(fs)
for feat in args.feats[1:]:
    d = os.path.join(args.d, feat)
    cfs = os.listdir(d)
    cfs = [f for f in cfs if f.endswith('.npy')]
    fs_set = fs_set.intersection(set(cfs))
print(len(fs_set))
fs = sorted(list(fs_set))

minmax_dict = {}
for feat_dir in args.feats:
    minmax_path = os.path.join(args.d, '%s_minmax.txt' % feat_dir)
    if os.path.exists(minmax_path):
        with open(minmax_path, 'r') as inf:
            lines = inf.readlines()
        l_list = lines[0].strip().split()
        l_list = [float(e) for e in l_list]
        minmax_dict[feat_dir] = (l_list[0], l_list[1], l_list[1]-l_list[0])

for f in tqdm(fs):
    cfeats = []
    for feat_dir in args.feats:
        p = os.path.join(args.d, feat_dir, f)
        cfeat = np.load(p)
        if len(cfeat.shape) == 1:
            cfeat = cfeat[:, np.newaxis]
        if feat_dir in minmax_dict:
            cmin = minmax_dict[feat_dir][0]
            crange = minmax_dict[feat_dir][1]
            cfeat = (cfeat-cmin)/crange
        cfeats.append(cfeat)
    min_len = min([len(cfeat) for cfeat in cfeats])
    cfeats = [cfeat[:min_len, :] for cfeat in cfeats]
    all_feat = np.concatenate(cfeats, axis=1)
    oup = os.path.join(oud, f)
    np.save(oup, all_feat)
