#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Make train-val-test split for MNGU0 EMA-to-Speech task."""

import os
import random
import numpy as np

from tqdm import tqdm


parentd = 'downloads/emadata'
rawd = os.path.join(parentd, 'cin_us_mngu0')
wavd = os.path.join(rawd, 'wav')
etcd = os.path.join(rawd, 'etc')
trainp = os.path.join(etcd, 'txt.done.data.train')
testp = os.path.join(etcd, 'txt.done.data.test')
with open(trainp, 'r') as inf:
    lines = inf.readlines()
random.Random(0).shuffle(lines)
num_val = 60
train_lines = lines[:-num_val]
val_lines = lines[-num_val:]
with open(testp, 'r') as inf:
    test_lines = inf.readlines()

train_lines = [l.strip() for l in train_lines]
val_lines = [l.strip() for l in val_lines]
test_lines = [l.strip() for l in test_lines]
train_fids = [l.split()[1] for l in train_lines]
val_fids = [l.split()[1] for l in val_lines]
test_fids = [l.split()[1] for l in test_lines]
train_fids = sorted(train_fids)
val_fids = sorted(val_fids)
test_fids = sorted(test_fids)

emad = os.path.join(rawd, 'nema')
actionsd = os.path.join(rawd, 'actions')
if not os.path.exists(actionsd):
    os.makedirs(actionsd)

train_dir = 'mngu0_train'
val_dir = 'mngu0_val'
test_dir = 'mngu0_test'
if not os.path.exists('data/%s' % train_dir):
    os.makedirs('data/%s' % train_dir)
if not os.path.exists('data/%s' % val_dir):
    os.makedirs('data/%s' % val_dir)
if not os.path.exists('data/%s' % test_dir):
    os.makedirs('data/%s' % test_dir)

new_train_fids = []
new_val_fids = []
new_test_fids = []
count = 0
with open('data/%s/feats.scp' % train_dir, 'w+') as ouf:
    for f in tqdm(train_fids):
        emap = os.path.join(emad, f+'.ema')
        with open(emap, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_list = [l.split() for l in lines]
        arr = np.array([[float(n) for n in l] for l in l_list])
        npp = os.path.join(actionsd, f+'.npy')
        if np.any(np.isnan(arr)):
            count += 1
        else:
            np.save(npp, arr)
            ouf.write('%s %s\n' % (f, npp))
            new_train_fids.append(f)
with open('data/%s/feats.scp' % val_dir, 'w+') as ouf:
    for f in tqdm(val_fids):
        emap = os.path.join(emad, f+'.ema')
        with open(emap, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_list = [l.split() for l in lines]
        arr = np.array([[float(n) for n in l] for l in l_list])
        npp = os.path.join(actionsd, f+'.npy')
        if np.any(np.isnan(arr)):
            count += 1
        else:
            np.save(npp, arr)
            ouf.write('%s %s\n' % (f, npp))
            new_val_fids.append(f)
with open('data/%s/feats.scp' % test_dir, 'w+') as ouf:
    for f in tqdm(test_fids):
        emap = os.path.join(emad, f+'.ema')
        with open(emap, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_list = [l.split() for l in lines]
        arr = np.array([[float(n) for n in l] for l in l_list])
        npp = os.path.join(actionsd, f+'.npy')
        if np.any(np.isnan(arr)):
            count += 1
        else:
            np.save(npp, arr)
            ouf.write('%s %s\n' % (f, npp))
            new_test_fids.append(f)
train_fids = new_train_fids
val_fids = new_val_fids
test_fids = new_test_fids
print(len(train_fids), len(val_fids), len(test_fids))

with open('data/%s/wav.scp' % train_dir, 'w+') as ouf:
    for f in train_fids:
        p = os.path.join(wavd, f+'.wav')
        ouf.write('%s %s\n' % (f, p))
with open('data/%s/wav.scp' % val_dir, 'w+') as ouf:
    for f in val_fids:
        p = os.path.join(wavd, f+'.wav')
        ouf.write('%s %s\n' % (f, p))
with open('data/%s/wav.scp' % test_dir, 'w+') as ouf:
    for f in test_fids:
        p = os.path.join(wavd, f+'.wav')
        ouf.write('%s %s\n' % (f, p))

spk = 'mngu0_s1'
with open('data/%s/utt2spk' % train_dir, 'w+') as ouf:
    for f in train_fids:
        ouf.write('%s %s\n' % (f, spk))
with open('data/%s/utt2spk' % val_dir, 'w+') as ouf:
    for f in val_fids:
        ouf.write('%s %s\n' % (f, spk))
with open('data/%s/utt2spk' % test_dir, 'w+') as ouf:
    for f in test_fids:
        ouf.write('%s %s\n' % (f, spk))

with open('data/%s/spk2utt' % train_dir, 'w+') as ouf:
    l = ' '.join(train_fids)
    ouf.write('%s %s\n' % (spk, l))
with open('data/%s/spk2utt' % val_dir, 'w+') as ouf:
    l = ' '.join(val_fids)
    ouf.write('%s %s\n' % (spk, l))
with open('data/%s/spk2utt' % test_dir, 'w+') as ouf:
    l = ' '.join(test_fids)
    ouf.write('%s %s\n' % (spk, l))
