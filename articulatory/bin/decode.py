#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained model."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

import articulatory.transforms

from articulatory.datasets import MelDataset, audio_mel_dataset
from articulatory.datasets import MelSCPDataset
from articulatory.datasets import ArtDataset
from articulatory.datasets import ArtSCPDataset, AudioSCPDataset, ArtSCPMultDataset
from articulatory.utils import load_model
from articulatory.utils import read_hdf5


def ar_loop(model, x, config, do_wsola=False, modality=None, generator2=False):
    '''
    Args:
        x: (art_len, num_feats)
    
    Return:
        signal: (audio_len,)
    '''
    if generator2:
        params_key = "generator2_params"
        w2a = False
    else:
        params_key = "generator_params"
        w2a = config["dataset_mode"] == 'w2a'
    audio_chunk_len = config["batch_max_steps"]
    if w2a:
        in_chunk_len = audio_chunk_len
        past_out_len = int(config[params_key]["ar_input"]/config[params_key]["out_channels"])
    else:
        in_chunk_len = int(audio_chunk_len/config["hop_size"])
        past_out_len = config[params_key]["ar_input"]
    if modality is not None:
        scale_factor = config["sampling_rate"]/config["hop_size"]*config["hop_sizes"][modality]/config["sampling_rates"][modality]
    if not do_wsola:
        # NOTE extra_art not supported
        ins = [x[i:i+in_chunk_len] for i in range(0, len(x), in_chunk_len)]
        if w2a and len(ins[-1]) < config["hop_size"]:
            ins = ins[:-1]
        prev_samples = torch.zeros((1, config[params_key]["out_channels"], past_out_len), dtype=x.dtype, device=x.device)
        outs = []
        
        for cin in ins: # a2w cin (in_chunk_len, num_feats)
            if len(cin.shape) == 1:
                cin = cin.unsqueeze(1)
            cin = cin.unsqueeze(0)  # a2w (1, in_chunk_len, num_feats)
            cin = cin.permute(0, 2, 1)  # a2w (1, num_feats, in_chunk_len)
            if modality is not None:
                new_cin = [None for _ in config[params_key]["in_list"]]
                cin = torch.nn.functional.interpolate(cin, scale_factor=scale_factor, mode='linear', align_corners=False)
                new_cin[modality] = cin
                cin = new_cin
            cout = model(cin, ar=prev_samples)  # a2w (1, 1, audio_chunk_length)
            if w2a:
                outs.append(cout[0].transpose(0, 1))
            else:
                outs.append(cout[0][0])
            if past_out_len <= audio_chunk_len:
                prev_samples = cout[:, :, -past_out_len:]
            else:
                prev_samples[:, :, :-in_chunk_len] = prev_samples[:, :, in_chunk_len:].clone()
                prev_samples[:, :, -in_chunk_len:] = cout
        out = torch.cat(outs, dim=0)  # w2a (seq_len, num_feats)
        return out
    else:
        # NOTE modality is not supported
        extra_art = config[params_key]["extra_art"]
        assert in_chunk_len % 2 == 0
        ins = [x[i:i+in_chunk_len+int(extra_art)] for i in range(0, len(x), int(in_chunk_len/2))]
        prev_samples = torch.zeros((1, 1, past_out_len), dtype=x.dtype, device=x.device)
        outs = []
        for art_i, art in enumerate(ins): # art (in_chunk_len, num_feats)
            art = art.unsqueeze(0)  # (1, in_chunk_len, num_feats)
            art = art.permute(0, 2, 1)  # (1, num_feats, in_chunk_len)
            signal = model(art, ar=prev_samples)  # (1, 1, audio_chunk_length)
            outs.append(signal[0][0])
            if art_i < len(ins)-1:
                prev_samples = signal[:, :, int(audio_chunk_len/2)-past_out_len:int(audio_chunk_len/2)]
                # print(signal.shape, in_chunk_len, prev_samples.shape, past_out_len)
                assert prev_samples.shape[2] == past_out_len
        return outs, ins


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained generator "
        "(See detail in articulatory/bin/decode.py)."
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
        "--dumpdir",
        default=None,
        type=str,
        help="directory including feature files. "
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
        "--normalize-before",
        default=False,
        action="store_true",
        help="whether to perform feature normalization before input to the model. "
        "if true, it assumes that the feature is de-normalized. this is useful when "
        "text2mel model and vocoder use different feature statistics.",
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

    # check arguments
    if (args.feats_scp is not None and args.dumpdir is not None) or (
        args.feats_scp is None and args.dumpdir is None
    ):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    if "dataset_mode" not in config:
        dataset_mode = 'default'
        config["dataset_mode"] = dataset_mode
    else:
        dataset_mode = config["dataset_mode"]
    if "transform" not in config:
        transform = None
    else:
        transform = config["transform"]
    input_transform = config.get("input_transform", transform)
    if input_transform is not None:
        input_transform = getattr(articulatory.transforms, input_transform)
    if dataset_mode == 'default' or dataset_mode == 'm2w':
        if args.dumpdir is not None:
            if config["format"] == "hdf5":
                mel_query = "*.h5"
                mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
            elif config["format"] == "npy":
                mel_query = "*-feats.npy"
                mel_load_fn = np.load
            else:
                raise ValueError("Support only hdf5 or npy format.")
            dataset = MelDataset(
                args.dumpdir,
                mel_query=mel_query,
                mel_load_fn=mel_load_fn,
                return_utt_id=True,
            )
        else:
            dataset = MelSCPDataset(
                feats_scp=args.feats_scp,
                return_utt_id=True,
            )
    elif dataset_mode == 'a2w' or dataset_mode == 'a2w_pcd' or dataset_mode == 'art' \
            or dataset_mode == 'ph2m' or dataset_mode == 'ph2a' or dataset_mode == 'a2m':
        if args.dumpdir is not None:
            if config["format"] == "hdf5":
                mel_query = "*.h5"
                mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
            elif config["format"] == "npy":
                mel_query = "*-feats.npy"
                mel_load_fn = np.load
            else:
                raise ValueError("Support only hdf5 or npy format.")
            dataset = ArtDataset(
                args.dumpdir,
                mel_query=mel_query,
                mel_load_fn=mel_load_fn,
                return_utt_id=True,
                transform=transform,
            )
        else:
            dataset = ArtSCPDataset(
                feats_scp=args.feats_scp,
                return_utt_id=True,
                transform=transform,
                input_transform=input_transform,
            )
    elif dataset_mode == 'a2w_mult':
        dataset = ArtSCPMultDataset(
            feats_scp=args.feats_scp,
            return_utt_id=True,
            transform=transform,
        )
    elif dataset_mode == 'w2a':
        dataset = AudioSCPDataset(
            wav_scp=args.feats_scp,
            return_utt_id=True
        )
    else:
        raise ValueError("dataset_mode %s not supported." % dataset_mode)
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    if args.normalize_before:
        assert hasattr(model, "mean"), "Feature stats are not registered."
        assert hasattr(model, "scale"), "Feature stats are not registered."
    model.remove_weight_norm()
    model = model.eval().to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

    do_wsola = "wsola" in config and config["wsola"]

    # start generation
    total_rtf = 0.0
    times = []
    input_lens = []
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, tup in enumerate(pbar, 1):
            if len(tup) == 2:
                utt_id, c = tup
                modality = None
            else:
                utt_id, c, modality = tup
            # generate
            c = torch.tensor(c, dtype=torch.float).to(device)
            input_lens.append(len(c))
            start = time.time()

            # save as PCM 16 bit wav file
            if dataset_mode == 'default' or dataset_mode == 'a2w' or dataset_mode == 'a2w_pcd' or dataset_mode == 'a2w_mult' or dataset_mode == 'm2w':
                if "use_ar" in config["generator_params"] and config["generator_params"]["use_ar"]:
                    t0 = time.time()
                    y = ar_loop(model, c, config, do_wsola=do_wsola, modality=modality)
                    t1 = time.time()
                else:
                    # NOTE modality not supported
                    t0 = time.time()
                    y = model.inference(c, normalize_before=args.normalize_before).view(-1)
                    t1 = time.time()
                if not do_wsola:
                    rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
                    pbar.set_postfix({"RTF": rtf})
                    total_rtf += rtf
                    sf.write(
                        os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                        y.cpu().numpy(),
                        config["sampling_rate"],
                        "PCM_16",
                    )
                else:
                    signals, arts = y
                    for cyi, cy in enumerate(signals):
                        rtf = (time.time() - start) / (len(cy) / config["sampling_rate"])
                        pbar.set_postfix({"RTF": rtf})
                        total_rtf += rtf
                        sf.write(
                            os.path.join(config["outdir"], "%s_%d_gen.wav" % (utt_id, cyi)),
                            cy.cpu().numpy(),
                            config["sampling_rate"],
                            "PCM_16",
                        )
                        np.save(os.path.join(config["outdir"], "%s_%d.npy" % (utt_id, cyi)), arts[cyi].cpu().numpy())
            elif dataset_mode == 'art' or dataset_mode == 'w2a' or dataset_mode == 'ph2m' or dataset_mode =='ph2a' or dataset_mode == 'a2m':
                if "use_ar" in config["generator_params"] and config["generator_params"]["use_ar"]:
                    t0 = time.time()
                    y = ar_loop(model, c, config, do_wsola=do_wsola, modality=modality)
                    t1 = time.time()
                else:
                    t0 = time.time()
                    # c is 1 dim if ph, 2 dim if a
                    if dataset_mode == 'ph2m' or dataset_mode == 'ph2a':
                        c = c.long()
                    y = model.inference(c, normalize_before=args.normalize_before)
                    t1 = time.time()
                np.save(os.path.join(config["outdir"], f"{utt_id}_gen.npy"), y.cpu().numpy())
            times.append(t1-t0)
    print('avg time:', np.mean(times))
    print('avg input len:', np.mean(input_lens))

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
