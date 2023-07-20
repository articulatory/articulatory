#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

"""Predict waveforms with trained model."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from articulatory.bin.decode import ar_loop
from articulatory.utils import load_model

def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained generator "
        "(See detail in ats/bin/decode.py)."
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file. "
        "you need to specify either feats-scp or dumpdir.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    with open(args.feats_scp, 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    fids = []
    featps = []
    for l in lines:
        l_list = l.split()
        fid = l_list[0]
        featp = l_list[1]
        fids.append(fid)
        featps.append(featp)

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    # if args.normalize_before:
    #     assert hasattr(model, "mean"), "Feature stats are not registered."
    #     assert hasattr(model, "scale"), "Feature stats are not registered."
    model.remove_weight_norm()
    model = model.eval().to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

    # start generation
    times = []
    input_lens = []
    with torch.no_grad():
        for fid, featp in tqdm(zip(fids, featps), total=len(fids)):
            c = np.load(featp, allow_pickle=True)
            # generate
            c = torch.tensor(c, dtype=torch.float).to(device)
            input_lens.append(len(c))
            if c.shape[0] > 250:
                if "use_ar" in config["generator_params"] and config["generator_params"]["use_ar"]:
                    y = ar_loop(model, c, config)
                else:
                    if len(c.shape) == 1:
                        c = c.long()
                    y = model.inference(c)
                sf.write(os.path.join(config["outdir"], fid+'.wav'), y.cpu().numpy(), config["sampling_rate"])

if __name__ == "__main__":
    main()
