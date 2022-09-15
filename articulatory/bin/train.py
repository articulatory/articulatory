#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Training script."""

import argparse
import functools
import logging
import math
import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import articulatory
import articulatory.models
import articulatory.optimizers
import articulatory.samplers
import articulatory.transforms

from articulatory.datasets import MelArtDataset, SpeechDataset, WavArtMultDataset
from articulatory.layers import PQMF
from articulatory.losses import DiscriminatorAdversarialLoss
from articulatory.losses import FeatureMatchLoss
from articulatory.losses import GeneratorAdversarialLoss
from articulatory.losses import MelSpectrogramLoss
from articulatory.losses import MultiResolutionSTFTLoss
# from articulatory.losses import InterLoss
from articulatory.utils import read_hdf5

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def combine_fixed_length(tensor_list, length):
    '''
    Args:
        tensor_list: list

    Return:
        tensor with shape (batch_size, num_feats, T)
    '''
    total_length = sum(t.size(0) for t in tensor_list)
    if total_length % length != 0:
        pad_length = length - (total_length % length)
        tensor_list = list(tensor_list)  # copy
        tensor_list.append(torch.zeros(pad_length,*tensor_list[0].size()[1:], dtype=torch.float, device=tensor_list[0].device))
        total_length += pad_length
    tensor = torch.cat(tensor_list, 0)
    n = total_length // length
    combined = tensor.view(n, length, *tensor.size()[1:])
    combined = combined.permute(0, 2, 1)
    return combined


class Trainer(object):
    """Customized trainer module for training articulatory models."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.best_mel_loss = 1.0e6
        if self.config.get("use_pcd", False):
            self.interp_fn = functools.partial(
                    torch.nn.functional.interpolate,
                    size=self.config["batch_max_steps"],
                    mode='linear',
                    align_corners=False)

        self.use_ar = self.config["generator_params"].get("use_ar", False)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
            if "generator2_type" in self.config:
                state_dict["model"]["generator2"] = self.model["generator2"].module.state_dict(),
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }
            if "generator2_type" in self.config:
                state_dict["model"]["generator2"] = self.model["generator2"].state_dict(),

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False, checkpoint2_path=None):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint2_path is not None:
            state_dict2 = torch.load(checkpoint2_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(
                state_dict["model"]["generator"]
            )
            if checkpoint2_path is not None:
                self.model["generator2"].module.load_state_dict(
                    state_dict2["model"]["generator"]
                )
                self.model["discriminator"].module.load_state_dict(
                    state_dict2["model"]["discriminator"]
                )
            else:
                self.model["discriminator"].module.load_state_dict(
                    state_dict["model"]["discriminator"]
                )
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            if checkpoint2_path is not None:
                self.model["generator2"].load_state_dict(state_dict2["model"]["generator"])
                self.model["discriminator"].load_state_dict(
                    state_dict2["model"]["discriminator"]
                )
            else:
                self.model["discriminator"].load_state_dict(
                    state_dict["model"]["discriminator"]
                )
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            if checkpoint2_path is not None:
                self.optimizer["discriminator"].load_state_dict(
                    state_dict2["optimizer"]["discriminator"]
                )
            else:
                self.optimizer["discriminator"].load_state_dict(
                    state_dict["optimizer"]["discriminator"]
                )
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"]
            )
            if checkpoint2_path is not None:
                self.scheduler["discriminator"].load_state_dict(
                    state_dict2["scheduler"]["discriminator"]
                )
            else:
                self.scheduler["discriminator"].load_state_dict(
                    state_dict["scheduler"]["discriminator"]
                )

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x = batch['x']
        y = batch['y'].to(self.device)
        ar = None if 'ar' not in batch else batch['ar'].to(self.device)
        ar2 = None if 'ar2' not in batch else batch['ar2'].to(self.device)
        spk_id = None if 'spk_id' not in batch else batch['spk_id'].to(self.device)
        ph = None if 'ph' not in batch else batch['ph'].to(self.device)
        pitch = None if 'pitch' not in batch else batch['pitch'].to(self.device)
        periodicity = None if 'periodicity' not in batch else batch['periodicity'].to(self.device)
        new_x = []
        for x_ in x:
            new_x_ = None
            if isinstance(x_, list):  # eg in multimodal case
                new_x_ = [t.to(self.device) if t is not None else None for t in x_]
            else:
                new_x_ = x_.to(self.device)
            new_x.append(new_x_)
        x = tuple(new_x)
        if "generator2" in self.model:
            inter = y 
            y = x[0].detach().clone()

        #######################
        #      Generator      #
        #######################
        if self.steps > self.config.get("generator_train_start_steps", 0):
            if "generator2" in self.model:
                y_ = self.model["generator"](*x, spk_id=spk_id, ar=ar)
                inter_ = y_
                y_ = self.model["generator2"](inter_, spk_id=spk_id, ar=ar2, ph=ph)
                if self.config["use_ph_loss"]:
                    y_, ph_ = y_
            else:
                y_ = self.model["generator"](*x, spk_id=spk_id, ar=ar, ph=ph)
                if self.config["use_ph_loss"]:
                    y_, ph_ = y_
        
            # reconstruct the signal from multi-band signal
            if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
                y_mb_ = y_
                y_ = self.criterion["pqmf"].synthesis(y_mb_)

            # initialize
            gen_loss = 0.0

            # multi-resolution sfft loss
            if self.config["use_stft_loss"]:
                sc_loss, mag_loss = self.criterion["stft"](y_, y)
                gen_loss += sc_loss + mag_loss
                self.total_train_loss[
                    "train/spectral_convergence_loss"
                ] += sc_loss.item()
                self.total_train_loss[
                    "train/log_stft_magnitude_loss"
                ] += mag_loss.item()

            # subband multi-resolution stft loss
            if self.config["use_subband_stft_loss"]:
                gen_loss *= 0.5  # for balancing with subband stft loss
                y_mb = self.criterion["pqmf"].analysis(y)
                sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
                gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
                self.total_train_loss[
                    "train/sub_spectral_convergence_loss"
                ] += sub_sc_loss.item()
                self.total_train_loss[
                    "train/sub_log_stft_magnitude_loss"
                ] += sub_mag_loss.item()

            # mel spectrogram loss
            if self.config["use_mel_loss"]:
                mel_loss = self.criterion["mel"](y_, y)
                gen_loss += mel_loss
                self.total_train_loss["train/mel_loss"] += mel_loss.item()
            
            # inter loss
            if self.config["use_inter_loss"]:
                inter_loss = self.criterion["inter"](inter_, inter)
                gen_loss += inter_loss
                self.total_train_loss["train/inter_loss"] += inter_loss.item()

            # weighting aux loss
            gen_loss *= self.config.get("lambda_aux", 1.0)

            # phoneme loss
            if self.config["use_ph_loss"]:
                ph_loss = self.criterion["ph"](ph_, ph)
                gen_loss += self.config["lambda_ph"] * ph_loss
                self.total_train_loss["train/ph_loss"] += ph_loss.item()

            # adversarial loss
            if self.config.get("use_pcd", False):
                pitch_interp = self.interp_fn(pitch)
                period_interp = self.interp_fn(periodicity)
                disc_y = torch.cat([y, pitch_interp, period_interp], dim=1)
                disc_y_ = torch.cat([y_, pitch_interp, period_interp], dim=1)
            else:
                if self.use_ar:
                    if ar2 is not None:
                        disc_y = torch.cat([ar2, y], dim=2)
                        disc_y_ = torch.cat([ar2, y_], dim=2)
                    else:
                        disc_y = torch.cat([ar, y], dim=2)
                        disc_y_ = torch.cat([ar, y_], dim=2)
                else:
                    disc_y = y
                    disc_y_ = y_
            if self.steps > self.config["discriminator_train_start_steps"]:
                p_ = self.model["discriminator"](disc_y_)
                adv_loss = self.criterion["gen_adv"](p_)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](disc_y)
                    fm_loss = self.criterion["feat_match"](p_, p)
                    self.total_train_loss[
                        "train/feature_matching_loss"
                    ] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

                # add adversarial loss to generator loss
                gen_loss += self.config["lambda_adv"] * adv_loss

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
            self.optimizer["generator"].zero_grad()
            gen_loss.backward()
            if self.config["generator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config["generator_grad_norm"],
                )
            self.optimizer["generator"].step()
            if self.config["generator_scheduler_type"] == "ReduceLROnPlateau":
                self.scheduler["generator"].step(gen_loss)
            else: 
                self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                if "generator2" in self.model:
                    y_ = self.model["generator"](*x, spk_id=spk_id, ar=ar)
                    inter_ = y_
                    y_ = self.model["generator2"](inter_, spk_id=spk_id, ar=ar2, ph=ph)
                    if self.config["use_ph_loss"]:
                        y_, ph_ = y_
                else:
                    y_ = self.model["generator"](*x, spk_id=spk_id, ar=ar, ph=ph)
                    if self.config["use_ph_loss"]:
                        y_, ph_ = y_
            if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
                y_ = self.criterion["pqmf"].synthesis(y_)

            # discriminator loss
            if self.use_ar:
                if ar2 is not None:
                    disc_y = torch.cat([ar2, y], dim=2)
                    disc_y_ = torch.cat([ar2, y_], dim=2)
                else:
                    disc_y = torch.cat([ar, y], dim=2)
                    disc_y_ = torch.cat([ar, y_], dim=2)
            else:
                disc_y = y
                disc_y_ = y_
            p = self.model["discriminator"](disc_y)
            p_ = self.model["discriminator"](disc_y_.detach())
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"],
                )
            self.optimizer["discriminator"].step()
            if self.config["discriminator_scheduler_type"] == "ReduceLROnPlateau":
                self.scheduler["discriminator"].step(dis_loss)
            else: 
                self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x = batch['x']
        y = batch['y'].to(self.device)
        ar = None if 'ar' not in batch else batch['ar'].to(self.device)
        ar2 = None if 'ar2' not in batch else batch['ar2'].to(self.device)
        spk_id = None if 'spk_id' not in batch else batch['spk_id'].to(self.device)
        ph = None if 'ph' not in batch else batch['ph'].to(self.device)
        pitch = None if 'pitch' not in batch else batch['pitch'].to(self.device)
        periodicity = None if 'periodicity' not in batch else batch['periodicity'].to(self.device)
        new_x = []
        for x_ in x:
            new_x_ = None
            if isinstance(x_, list):  # eg in multimodal case
                new_x_ = [t.to(self.device) if t is not None else None for t in x_]
            else:
                new_x_ = x_.to(self.device)
            new_x.append(new_x_)
        x = tuple(new_x)
        if "generator2" in self.model:
            inter = y
            y = x[0].detach().clone()

        #######################
        #      Generator      #
        #######################
        if "generator2" in self.model:
            y_ = self.model["generator"](*x, spk_id=spk_id, ar=ar)
            inter_ = y_
            y_ = self.model["generator2"](inter_, spk_id=spk_id, ar=ar2, ph=ph)
            if self.config["use_ph_loss"]:
                y_, ph_ = y_
        else:
            y_ = self.model["generator"](*x, spk_id=spk_id, ar=ar, ph=ph)
            if self.config["use_ph_loss"]:
                y_, ph_ = y_
        if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        # initialize
        aux_loss = 0.0  # called gen_loss during training

        # multi-resolution stft loss
        if self.config["use_stft_loss"]:
            sc_loss, mag_loss = self.criterion["stft"](y_, y)
            aux_loss += sc_loss + mag_loss
            self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
            self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            aux_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_eval_loss[
                "eval/sub_spectral_convergence_loss"
            ] += sub_sc_loss.item()
            self.total_eval_loss[
                "eval/sub_log_stft_magnitude_loss"
            ] += sub_mag_loss.item()
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # mel spectrogram loss
        if self.config["use_mel_loss"]:
            mel_loss = self.criterion["mel"](y_, y)
            aux_loss += mel_loss
            self.total_eval_loss["eval/mel_loss"] += mel_loss.item()

        # inter loss
        if self.config["use_inter_loss"]:
            inter_loss = self.criterion["inter"](inter_, inter)
            aux_loss += inter_loss
            self.total_eval_loss["eval/inter_loss"] += inter_loss.item()

        # weighting stft loss
        aux_loss *= self.config.get("lambda_aux", 1.0)

        # phoneme loss
        if self.config["use_ph_loss"]:
            ph_loss = self.criterion["ph"](ph_, ph)
            aux_loss += self.config["lambda_ph"] * ph_loss
            self.total_eval_loss["eval/ph_loss"] += ph_loss.item()

        # adversarial loss
        if self.config.get("use_pcd", False):
            pitch_interp = self.interp_fn(pitch)
            period_interp = self.interp_fn(periodicity)
            disc_y = torch.cat([y, pitch_interp, period_interp], dim=1)
            disc_y_ = torch.cat([y_, pitch_interp, period_interp], dim=1)
        else:
            if self.use_ar:
                if ar2 is not None:
                    disc_y = torch.cat([ar2, y], dim=2)
                    disc_y_ = torch.cat([ar2, y_], dim=2)
                else:
                    disc_y = torch.cat([ar, y], dim=2)
                    disc_y_ = torch.cat([ar, y_], dim=2)
            else:
                disc_y = y
                disc_y_ = y_
        p_ = self.model["discriminator"](disc_y_)
        adv_loss = self.criterion["gen_adv"](p_)
        gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

        # feature matching loss
        if self.config["use_feat_match_loss"]:
            p = self.model["discriminator"](disc_y)
            fm_loss = self.criterion["feat_match"](p_, p)
            self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
            gen_loss += (
                self.config["lambda_adv"] * self.config["lambda_feat_match"] * fm_loss
            )

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](disc_y)
        p_ = self.model["discriminator"](disc_y_)

        # discriminator loss
        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._generate_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        if self.total_eval_loss["eval/mel_loss"] < self.best_mel_loss:
            best_mel_p = os.path.join(self.config["outdir"], "best_mel_step.txt")
            with open(best_mel_p, "w+") as ouf:
                ouf.write("%d\n" % self.steps)
            self.save_checkpoint(os.path.join(self.config["outdir"], "best_mel_ckpt.pkl"))
            self.best_mel_loss = self.total_eval_loss["eval/mel_loss"]

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # parse batch
        x_temp = batch['x']
        y_temp = batch['y'].to(self.device)
        ar_temp = None if 'ar' not in batch else batch['ar'].to(self.device)
        ar2_temp = None if 'ar2' not in batch else batch['ar2'].to(self.device)
        spk_id_temp = None if 'spk_id' not in batch else batch['spk_id'].to(self.device)
        ph_temp = None if 'ph' not in batch else batch['ph'].to(self.device)
        pitch_temp = None if 'pitch' not in batch else batch['pitch'].to(self.device)
        periodicity_temp = None if 'periodicity' not in batch else batch['periodicity'].to(self.device)
        new_x_temp = []
        for x_ in x_temp:
            new_x_ = None
            if isinstance(x_, list):  # eg in multimodal case
                new_x_ = [t.to(self.device) if t is not None else None for t in x_]
            else:
                new_x_ = x_.to(self.device)
            new_x_temp.append(new_x_)
        x_temp = tuple(new_x_temp)
        if "generator2" in self.model:
            inter_temp = y_temp
            y_temp = x_temp[0].detach().clone()

        # generate
        if "generator2" in self.model:
            y_temp_ = self.model["generator"](*x_temp, spk_id=spk_id_temp, ar=ar_temp)
            inter_temp_ = y_temp_
            y_temp_ = self.model["generator2"](inter_temp_, spk_id=spk_id_temp, ar=ar2_temp, ph=ph_temp)
            if self.config["use_ph_loss"]:
                y_temp_, ph_temp_ = y_temp_
        else:
            y_temp_ = self.model["generator"](*x_temp, spk_id=spk_id_temp, ar=ar_temp, ph=ph_temp)
            if self.config["use_ph_loss"]:
                y_temp_, ph_temp_ = y_temp_
        if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
            y_temp_ = self.criterion["pqmf"].synthesis(y_temp_)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_temp, y_temp_), 1):
            if y.shape[0] == 1:
                y = y[0]
            if y_.shape[0] == 1:
                y_ = y_[0]
            
            # convert to ndarray
            if len(y.shape) == 1:
                y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()
            else:
                y, y_ = y.cpu().numpy(), y_.cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            if len(y.shape) == 1:
                plt.plot(y)
            else: # (C, T')
                plt.imshow(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            if len(y_.shape) == 1:
                plt.plot(y_)
            else: # (C, T')
                plt.imshow(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            if len(y.shape) == 1:
                y = np.clip(y, -1, 1)
                y_ = np.clip(y_, -1, 1)
                sf.write(
                    figname.replace(".png", "_ref.wav"),
                    y,
                    self.config["sampling_rate"],
                    "PCM_16",
                )
                sf.write(
                    figname.replace(".png", "_gen.wav"),
                    y_,
                    self.config["sampling_rate"],
                    "PCM_16",
                )

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class CollaterMelArt(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_steps=20480,
        hop_size=256,
        aux_context_window=2,
        use_noise_input=False,
        ar_len=None,
        dataset_mode='a2m',
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.ar_len = ar_len
        self.dataset_mode = dataset_mode

        # set useful values in random cutting
        self.start_offset = aux_context_window
        self.end_offset = -(self.batch_max_frames + aux_context_window)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        # batch = [
        #     b for b in batch if len(b[0]) > self.mel_threshold
        # ]
        # print(len(batch))
        cs, arts = [b[0] for b in batch], [b[1] for b in batch]

        # make batch with random cut
        c_lengths = [len(c) for c in cs]
        start_frames = np.array(
            [
                np.random.randint(self.start_offset, cl + self.end_offset)
                for cl in c_lengths
            ]
        )
        c_starts = start_frames - self.aux_context_window
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window
        c_batch = [c[start:end] for c, start, end in zip(cs, c_starts, c_ends)]
        art_starts = c_starts
        art_ends = start_frames + self.batch_max_frames + self.aux_context_window
        art_batch = [art[start:end] for art, start, end in zip(arts, art_starts, art_ends)]

        # convert each batch to tensor, assume that each item in batch has the same length
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
        art_batch = torch.tensor(art_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')

        if self.ar_len is None:
            if self.dataset_mode == 'm2a':
                return {'x':(c_batch,), 'y':art_batch}
            else:
                return {'x':(art_batch,), 'y':c_batch}
        else:
            logging.error('ar not supported')
            exit()


class SpeechCollater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_steps=20480,
        hop_size=256,
        aux_context_window=0,
        use_noise_input=False,
        dataset_mode='a2w',
        use_spk_id=False,
        use_ph=False,
        config=None,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        assumes all mel lengths are > self.batch_max_frames + 2 * aux_context_window

        """
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size  # for mel and art
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.dataset_mode = dataset_mode
        self.use_ar = config["generator_params"].get("use_ar", False)
        if self.use_ar:
            self.ar_len = int(config["generator_params"].get("ar_input", 512)/config["generator_params"]["out_channels"])
            self.ar2_len = None
            if "generator2_params" in config:
                self.ar2_len = int(config["generator2_params"].get("ar_input", 512)/config["generator2_params"]["out_channels"])
            elif self.dataset_mode == 'a2w' or self.dataset_mode == 'm2w':
                self.ar2_len = self.ar_len
                self.ar_len = None
        else:
            self.ar_len = None
            self.ar2_len = None
        self.package_mode = config.get("package_mode", "random_window")
        if self.package_mode == "pad":
            self.pad_audio = config.get("pad_audio", 0.0)
            self.pad_art = config.get("pad_art", 0.0)
            self.pad_ph = config.get("pad_ph", 0)
        self.use_spk_id = use_spk_id
        self.use_ph = use_ph

        # set useful values in random cutting
        self.start_offset = aux_context_window  # 0, only used for selecting start idx
        self.end_offset = -(self.batch_max_frames + aux_context_window)
            # -self.batch_max_frames; only used for selecting start idx

        self.config = config
        if self.config is not None:
            self.audio_seq_len = self.config["batch_max_steps"]
            self.art_seq_len = int(self.audio_seq_len/self.config["hop_size"])

        if self.dataset_mode == 'a2w':
            self.x_key = 'art'
            self.y_key = 'audio'
            self.use_audio = True
            self.use_mel = False
            self.use_art = True
        elif self.dataset_mode == 'w2a':
            self.x_key = 'audio'
            self.y_key = 'art'
            self.use_audio = True
            self.use_mel = False
            self.use_art = True
        elif self.dataset_mode == 'ph2a':
            self.x_key = 'ph'
            self.y_key = 'art'
            self.use_audio = False
            self.use_mel = False
            self.use_art = True
        elif self.dataset_mode == 'ph2m':
            self.x_key = 'ph'
            self.y_key = 'mel'
            self.use_audio = False
            self.use_mel = True
            self.use_art = False
        elif self.dataset_mode == 'm2w':
            self.x_key = 'mel'
            self.y_key = 'audio'
            self.use_audio = True
            self.use_mel = True
            self.use_art = False
        else:
            logging.error('dataset_mode %s not supported' % self.dataset_mode)
            exit()

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        audios = []
        arts = []
        spk_ids = []
        phs = []
        mels = []
        for d in batch:
            audio = d['audio']
            art = d['art']
            art = art[:int(len(audio)/self.hop_size)]
            if len(art) + self.end_offset > self.start_offset:
                audios.append(audio)
                arts.append(art)
                if self.use_spk_id:
                    spk_ids.append(d['spk_id'])
                if self.use_ph:
                    phs.append(d['ph'])
                if self.use_mel:
                    mels.append(d['mel'])
        batch = {}
        if self.use_spk_id:
            batch['spk_id'] = torch.tensor(spk_ids, dtype=torch.long)
        if self.package_mode == 'window':
            audios = [torch.from_numpy(t).float()[:len(arts[i])*self.hop_size] for i, t in enumerate(audios)]
            arts = [torch.from_numpy(t).float() for t in arts]
            audio_batch = combine_fixed_length([t.to(self.device, non_blocking=True).unsqueeze(1) for t in audios], self.audio_seq_len)
            art_batch = combine_fixed_length([t.to(self.device, non_blocking=True) for t in arts], self.art_seq_len)
            if self.ar_len is not None:
                logging.error('wav_starts and art_starts unimplemented')
                exit()
            if self.use_ph:
                batch['ph'] = combine_fixed_length([t.to(self.device, non_blocking=True) for t in phs], self.art_seq_len).long()
        elif self.package_mode == 'random_window':
            # make batch with random cut
            c_lengths = [len(c) for c in arts]
            # NOTE assumes that all c_lengths >= self.batch_max_frames
            start_frames = np.array([np.random.randint(self.start_offset, cl+self.end_offset) for cl in c_lengths])
            wav_starts = start_frames * self.hop_size
            wav_ends = wav_starts + self.batch_max_steps
            art_starts = start_frames - self.aux_context_window
                # start_frames
            art_ends = start_frames + self.batch_max_frames + self.aux_context_window
                # start_frames + self.batch_max_frames
            audio_batch = [audio[start:end] for audio, start, end in zip(audios, wav_starts, wav_ends)]
            art_batch = [art[start:end] for art, start, end in zip(arts, art_starts, art_ends)]
            # convert each batch to tensor, assume that each item in batch has the same length
            audio_batch = np.stack(audio_batch, axis=0)
            audio_batch = torch.tensor(audio_batch, dtype=torch.float)
            audio_batch = audio_batch.unsqueeze(1)  # (B, 1, T)
            art_batch = np.stack(art_batch, axis=0)
            art_batch = torch.tensor(art_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
            if self.use_ph:
                ph_batch = [ph[start:end] for ph, start, end in zip(phs, art_starts, art_ends)]
                ph_batch = np.stack(ph_batch, axis=0)
                batch['ph'] = torch.tensor(ph_batch, dtype=torch.long)
            if self.use_mel:
                mel_batch = [mel[start:end] for mel, start, end in zip(mels, art_starts, art_ends)]
                mel_batch = np.stack(mel_batch, axis=0)
                batch['mel'] = torch.tensor(mel_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
        elif self.package_mode == 'pad':
            audios = [torch.from_numpy(t).float()[:len(arts[i])*self.hop_size] for i, t in enumerate(audios)]
            arts = [torch.from_numpy(t).float() for t in arts]
            max_art_len = max([len(t) for t in arts])
            max_audio_len = max_art_len*self.hop_size
            new_audios = []
            for t in audios:
                pad_length = max_audio_len-len(t)
                cpad = torch.ones(pad_length, *t.size()[1:], dtype=torch.float, device=t.device)*self.pad_audio
                new_t = torch.cat([t, cpad], 0)
                new_audios.append(new_t)
            audio_batch = torch.stack(new_audios).unsqueeze(1)  # (B, 1, T)
            new_arts = []
            for t in arts:
                pad_length = max_art_len-len(t)
                cpad = torch.ones(pad_length, *t.size()[1:], dtype=torch.float, device=t.device)*self.pad_art
                new_t = torch.cat([t, cpad], 0)
                new_arts.append(new_t)
            art_batch = torch.stack(new_arts).transpose(2, 1)  # (B, C, T')
            if self.use_ph:
                ph_batch = [torch.from_numpy(ph[:len(a)]).long() for ph, a in zip(phs, arts)]
                new_phs = []
                for t in ph_batch:
                    pad_length = max_art_len-len(t)
                    cpad = torch.ones(pad_length, *t.size()[1:], dtype=torch.long, device=t.device)*self.pad_ph
                    new_t = torch.cat([t, cpad], 0)
                    new_phs.append(new_t)
                batch['ph'] = torch.stack(new_phs)
        if self.use_audio:
            batch['audio'] = audio_batch
        if self.use_art:
            batch['art'] = art_batch
        batch['x'] = (batch[self.x_key],)
        batch['y'] = batch[self.y_key]
        if self.use_ar:
            if self.ar_len is not None:
                ar_batch = []
                for art, start in zip(arts, art_starts):
                    if start >= self.ar_len:
                        ar = art[start-self.ar_len:start]
                    else:
                        ar = art[:start]  # (T, channels)
                        ar = np.pad(ar, ((self.ar_len-len(ar),0), (0,0)), mode='constant', constant_values=0)
                    ar_batch.append(ar)
                ar_batch = np.stack(ar_batch, axis=0)
                ar_batch = torch.tensor(ar_batch, dtype=torch.float).transpose(2, 1)  # (B, channels, T_ar)
            if self.ar2_len is not None:  # NOTE dataset_mode == 'ph2a' AR not supported
                ar2_batch = []
                for wav, start in zip(audios, wav_starts):
                    if start >= self.ar2_len:
                        ar = wav[start-self.ar2_len:start]
                    else:
                        ar = wav[:start]
                        ar = np.pad(ar, (self.ar2_len-len(ar), 0), 'constant', constant_values=0)
                    ar2_batch.append(ar)
                ar2_batch = np.stack(ar2_batch, axis=0)
                ar2_batch = torch.tensor(ar2_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T_ar)
            if 'generator2_type' in self.config:
                batch['ar'] = ar_batch
                batch['ar2'] = ar2_batch
            else:
                batch['ar'] = ar2_batch if self.ar_len is None else ar_batch
        return batch


class SpeechCollaterMult(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_steps=20480,
        hop_size=256,
        aux_context_window=0,
        use_noise_input=False,
        ar_len=None,
        random_window=True,
        dataset_mode='a2w',
        hop_sizes=None,
        sampling_rate=None,
        sampling_rates=None,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        assumes all mel lengths are > self.batch_max_frames + 2 * aux_context_window
        """
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size  # for art
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.ar_len = ar_len
        self.random_window = random_window  # NOTE only supports true
        self.dataset_mode = dataset_mode

        self.hop_sizes = hop_sizes
        self.sampling_rate = sampling_rate
        self.sampling_rates = sampling_rates
        self.rem_art_coefs = [sr/self.sampling_rate/h for h, sr in zip(self.hop_sizes, self.sampling_rates)]

        # set useful values in random cutting
        self.start_offset = aux_context_window # 0, only used for selecting start idx
        self.end_offset = -(self.batch_max_frames + aux_context_window)
            # -self.batch_max_frames; only used for selecting start idx

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        audios = [[] for _ in self.hop_sizes]
        arts = [[] for _ in self.hop_sizes]
        for b in batch:
            audio = b[0]
            art = b[1]
            modality_i = b[2]
            rem_audio = len(audio) % self.hop_size
            if rem_audio > 0:
                audio = audio[:-rem_audio]
                rem_art = round(rem_audio*self.rem_art_coefs[modality_i])
                if rem_art > 0:
                    art = art[:-rem_art]  # (seq_len, num_feats)
            art = torch.from_numpy(art).float().unsqueeze(0).transpose(2, 1)  # (1, num_feats, seq_len)
            new_seq_len = len(audio) // self.hop_size
            art = F.interpolate(art, size=new_seq_len, mode='linear', align_corners=False)  # (1, num_feats, new_seq_len)
            art = art[0].transpose(1, 0)  # (new_seq_len, num_feats)
            audios[modality_i].append(audio)
            arts[modality_i].append(art)

        new_audios = []
        for l in audios:
            new_audios += l
        audios = new_audios

        # make batch with random cut
        art_lengths = []
        for art_list in arts:
            art_lengths += [len(art) for art in art_list]
        start_frames = np.array([np.random.randint(self.start_offset, art_len + self.end_offset) for art_len in art_lengths])
        y_starts = start_frames * self.hop_size   # waveform
        y_ends = y_starts + self.batch_max_steps  # waveform
        y_batch = [y[start:end] for y, start, end in zip(audios, y_starts, y_ends)]
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)

        art_starts = start_frames - self.aux_context_window  # i.e., start_frames
        art_ends = start_frames + self.batch_max_frames + self.aux_context_window
        art_batch = [] # [art[start:end] for art, start, end in zip(arts, art_starts, art_ends)]
        i = 0
        for art_list in arts:
            if len(art_list) == 0:
                art_tens = None
            else:
                art_tens = []
                for art in art_list:
                    art_tens.append(art[art_starts[i]:art_ends[i]])
                    i += 1
                art_tens = torch.stack(art_tens, dim=0).transpose(2, 1)  # (B, C, T')
            art_batch.append(art_tens)

        if self.ar_len is None:
            if self.dataset_mode == 'a2w':
                return (art_batch,), y_batch
            else:
                return (y_batch,), art_batch
        else:
            if self.dataset_mode == 'a2w' or self.dataset_mode == 'a2w_mult':
                ar_batch = []
                for x, start in zip(audios, y_starts):
                    if start >= self.ar_len:
                        ar = x[start-self.ar_len:start]
                    else:
                        ar = x[:start]
                        ar = np.pad(ar, (self.ar_len-len(ar), 0), 'constant', constant_values=0)
                    ar_batch.append(ar)
                ar_batch = torch.tensor(ar_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T_ar)
                return (art_batch,), y_batch, ar_batch
            else:
                logging.error('%s not supported for ar case' % self.dataset_mode)
                exit()


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_steps=20480,
        hop_size=256,
        aux_context_window=2,
        use_noise_input=False,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input

        # set useful values in random cutting
        self.start_offset = aux_context_window
        self.end_offset = -(self.batch_max_frames + aux_context_window)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        # check length
        batch = [
            self._adjust_length(*b) for b in batch if len(b[1]) > self.mel_threshold
        ]
        xs, cs = [b[0] for b in batch], [b[1] for b in batch]

        # make batch with random cut
        c_lengths = [len(c) for c in cs]
        start_frames = np.array(
            [
                np.random.randint(self.start_offset, cl + self.end_offset)
                for cl in c_lengths
            ]
        )
        x_starts = start_frames * self.hop_size
        x_ends = x_starts + self.batch_max_steps
        c_starts = start_frames - self.aux_context_window
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window
        y_batch = [x[start:end] for x, start, end in zip(xs, x_starts, x_ends)]
        c_batch = [c[start:end] for c, start, end in zip(cs, c_starts, c_ends)]

        # convert each batch to tensor, assume that each item in batch has the same length
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')

        # make input noise signal batch tensor
        if self.use_noise_input:
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            return (z_batch, c_batch), y_batch
        else:
            return (c_batch,), y_batch

    def _adjust_length(self, x, c):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        return x, c


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train articulatory model (See detail in articulatory/bin/train.py)."
    )
    parser.add_argument(
        "--train-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for training. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--train-feats-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--train-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for training.",
    )
    parser.add_argument(
        "--train-dumpdir",
        default=None,
        type=str,
        help="directory including training data. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--train-dumpdirs",
        default=None,
        type=str,
        help="directory including training data. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--dev-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for validation. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--dev-feats-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--dev-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for validation.",
    )
    parser.add_argument(
        "--dev-dumpdir",
        default=None,
        type=str,
        help="directory including development data. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--dev-dumpdirs",
        default=None,
        type=str,
        help="directory including development data. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--pretrain2",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if (args.train_feats_scp is not None and args.train_dumpdir is not None) or (
        args.train_feats_scp is None and args.train_dumpdir is None
    ):
        raise ValueError("Please specify either --train-dumpdir or --train-*-scp.")
    if (args.dev_feats_scp is not None and args.dev_dumpdir is not None) or (
        args.dev_feats_scp is None and args.dev_dumpdir is None
    ):
        raise ValueError("Please specify either --dev-dumpdir or --dev-*-scp.")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = articulatory.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["batch_max_steps"] // config[
            "hop_size"
        ] + 2 * config["generator_params"].get("aux_context_window", 0)
    else:
        mel_length_threshold = None
    if args.train_wav_scp is None or args.dev_wav_scp is None:
        if config["format"] == "hdf5":
            audio_query, mel_query = "*.h5", "*.h5"
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
        elif config["format"] == "npy":
            audio_query, mel_query = "*-wave.npy", "*-feats.npy"
            audio_load_fn = np.load
            mel_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
    if "dataset_mode" not in config:
        dataset_mode = 'default'
    else:
        dataset_mode = config["dataset_mode"]
    if "transform" not in config:
        transform = None
    else:
        transform = config["transform"]
    input_transform = config.get("input_transform", transform)
    if input_transform is not None:
        input_transform = getattr(articulatory.transforms, input_transform)
    output_transform = config.get("output_transform", transform)
    if output_transform is not None:
        output_transform = getattr(articulatory.transforms, output_transform)
    if dataset_mode == 'art' or dataset_mode == 'a2m' or dataset_mode == 'm2a':
        assert args.train_dumpdir is not None and args.dev_dumpdir is not None
        train_dataset = MelArtDataset(
            root_dir=args.train_dumpdir, audio_query=audio_query, mel_query=mel_query,
            audio_load_fn=audio_load_fn, mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),
            transform=transform,
        )
        dev_dataset = MelArtDataset(
            root_dir=args.dev_dumpdir, audio_query=audio_query, mel_query=mel_query,
            audio_load_fn=audio_load_fn, mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),
            transform=transform,
        )
        if not config["generator_params"].get("use_ar", False):
            ar_len = None
        else:
            ar_len = int(config["generator_params"]["ar_input"]/config["generator_params"]["out_channels"])
        collater = CollaterMelArt(
            batch_max_steps=config["batch_max_steps"], hop_size=config["hop_size"],
            aux_context_window=config["generator_params"].get("aux_context_window", 0),
            use_noise_input=config.get("generator_type", "ParallelWaveGANGenerator")
            in ["ParallelWaveGANGenerator"],
            ar_len=ar_len,
            dataset_mode=dataset_mode
        )
        train_collater = collater
        dev_collater = collater
    elif dataset_mode == 'a2w' or dataset_mode == 'w2a' or dataset_mode == 'ph2a' or dataset_mode == 'ph2m' or dataset_mode == 'm2w':
        assert args.train_dumpdir is not None and args.dev_dumpdir is not None
        use_spk_id = config["generator_params"].get("use_spk_id", False)
        use_ph = config["generator_params"].get("use_ph", False) or config["generator_params"].get("use_ph_loss", False) \
                    or dataset_mode == 'ph2a' or dataset_mode == 'ph2m'
        train_dataset = SpeechDataset(
            root_dir=args.train_dumpdir, audio_query=audio_query, audio_load_fn=audio_load_fn, mel_query=mel_query, mel_load_fn=mel_load_fn,
            allow_cache=config.get("allow_cache", False), transform=transform,
            input_transform=input_transform, output_transform=output_transform,
            use_spk_id=use_spk_id, use_ph=use_ph, dataset_mode=dataset_mode,
        )
        if use_spk_id:
            assert len(train_dataset.spks) == config["generator_params"]["num_spk"]
        dev_dataset = SpeechDataset(
            root_dir=args.dev_dumpdir, audio_query=audio_query, audio_load_fn=audio_load_fn, mel_query=mel_query, mel_load_fn=mel_load_fn,
            allow_cache=config.get("allow_cache", False), transform=transform,
            input_transform=input_transform, output_transform=output_transform,
            use_spk_id=use_spk_id, use_ph=use_ph, spks=train_dataset.spks, dataset_mode=dataset_mode,
        )
        train_collater = SpeechCollater(
            batch_max_steps=config["batch_max_steps"], hop_size=config["hop_size"],
            aux_context_window=config["generator_params"].get("aux_context_window", 0), # so 0
            use_noise_input=config.get("generator_type", "ParallelWaveGANGenerator") in ["ParallelWaveGANGenerator"],
            dataset_mode=dataset_mode, use_spk_id=use_spk_id, use_ph=use_ph, config=config,
        )
        dev_collater = SpeechCollater(
            batch_max_steps=config["batch_max_steps"], hop_size=config["hop_size"],
            aux_context_window=config["generator_params"].get("aux_context_window", 0), # so 0
            use_noise_input=config.get("generator_type", "ParallelWaveGANGenerator") in ["ParallelWaveGANGenerator"],
            dataset_mode=dataset_mode, use_spk_id=use_spk_id, use_ph=use_ph, config=config,
        )  # NOTE package_mode originally was always random_window for dev
    elif dataset_mode == 'a2w_mult':
        assert args.train_dumpdirs is not None
        assert args.dev_dumpdirs is not None
        train_dataset = WavArtMultDataset(
            root_dirs=args.train_dumpdirs.split(), audio_query=audio_query, audio_load_fn=audio_load_fn,
            allow_cache=config.get("allow_cache", False), transform=transform,
            sampling_rate=config["sampling_rate"], sampling_rates=config["sampling_rates"],
            ignore_modalities=config["ignore_modalities"],
        )
        dev_dataset = WavArtMultDataset(
            root_dirs=args.dev_dumpdirs.split(), audio_query=audio_query, audio_load_fn=audio_load_fn,
            allow_cache=config.get("allow_cache", False), transform=transform,
            sampling_rate=config["sampling_rate"], sampling_rates=config["sampling_rates"],
            ignore_modalities=config["ignore_modalities"],
        )
        collater = SpeechCollaterMult(
            batch_max_steps=config["batch_max_steps"], hop_size=config["hop_size"],
            aux_context_window=config["generator_params"].get("aux_context_window", 0), # so 0
            use_noise_input=config.get("generator_type", "ParallelWaveGANGenerator") in ["ParallelWaveGANGenerator"],
            ar_len=None if not config["generator_params"].get("use_ar", False) else config["generator_params"].get("ar_input", 512),
            dataset_mode=dataset_mode,
            hop_sizes=config["hop_sizes"],
            sampling_rate=config["sampling_rate"],
            sampling_rates=config["sampling_rates"],
        )
        train_collater = collater
        dev_collater = collater
    else:
        raise ValueError("dataset_mode %s not supported." % dataset_mode)

    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"The number of development files = {len(dev_dataset)}.")

    dataset = {"train": train_dataset, "dev": dev_dataset}

    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(dataset=dataset["train"], num_replicas=args.world_size, rank=args.rank, shuffle=True)
        sampler["dev"] = DistributedSampler(dataset=dataset["dev"], num_replicas=args.world_size, rank=args.rank, shuffle=False)

    data_loader = {
        "dev": DataLoader(
            dataset=dataset["dev"], shuffle=False if args.distributed else True, collate_fn=dev_collater,
            batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=sampler["dev"], pin_memory=config["pin_memory"],
        ),
    }

    batch_sampler_type = config.get("batch_sampler_type", "None")
    batch_sampler = {"train": None, "dev": None}
    if batch_sampler_type != "None":
        train_audio_lens_path = os.path.join(args.train_dumpdir, 'train_audio_lens.npy')
        if os.path.exists(train_audio_lens_path):
            train_audio_lens = np.load(train_audio_lens_path)
        else:
            train_audio_lens = []
            for audio, art in train_dataset:
                train_audio_lens.append(len(audio))
            train_audio_lens = np.array(train_audio_lens)
            np.save(train_audio_lens_path, train_audio_lens)
        batch_sampler_class = getattr(articulatory.samplers, batch_sampler_type)
        batch_sampler["train"] = batch_sampler_class(train_audio_lens, **config["batch_sampler_params"])
        data_loader["train"] = DataLoader(
            dataset=dataset["train"], collate_fn=train_collater,
            num_workers=config["num_workers"], batch_sampler=batch_sampler["train"], pin_memory=config["pin_memory"],
        )
    else:
        data_loader["train"] = DataLoader(
            dataset=dataset["train"], shuffle=False if args.distributed else True, collate_fn=train_collater,
            batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=sampler["train"], pin_memory=config["pin_memory"],
        )

    # define models
    generator_class = getattr(
        articulatory.models,
        # keep compatibility
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    discriminator_class = getattr(
        articulatory.models,
        # keep compatibility
        config.get("discriminator_type", "ParallelWaveGANDiscriminator"),
    )
    model = {
        "generator": generator_class(**config["generator_params"]).to(device),
        "discriminator": discriminator_class(**config["discriminator_params"]).to(device),
    }
    if "generator2_type" in config:
        generator2_class = getattr(
            articulatory.models,
            # keep compatibility
            config.get("generator2_type", "ParallelWaveGANGenerator"),
        )
        model["generator2"] = generator2_class(**config["generator2_params"]).to(device)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    logging.info("generator params = %s." % count_parameters(model["generator"]))

    # define criterions
    criterion = {
        "gen_adv": GeneratorAdversarialLoss(
            # keep compatibility
            **config.get("generator_adv_loss_params", {})
        ).to(device),
        "dis_adv": DiscriminatorAdversarialLoss(
            # keep compatibility
            **config.get("discriminator_adv_loss_params", {})
        ).to(device),
    }
    if config.get("use_stft_loss", True):  # keep compatibility
        config["use_stft_loss"] = True
        criterion["stft"] = MultiResolutionSTFTLoss(
            **config["stft_loss_params"],
        ).to(device)
    if config.get("use_subband_stft_loss", False):  # keep compatibility
        assert config["generator_params"]["out_channels"] > 1
        criterion["sub_stft"] = MultiResolutionSTFTLoss(
            **config["subband_stft_loss_params"],
        ).to(device)
    else:
        config["use_subband_stft_loss"] = False
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["feat_match"] = FeatureMatchLoss(
            # keep compatibility
            **config.get("feat_match_loss_params", {}),
        ).to(device)
    else:
        config["use_feat_match_loss"] = False
    if config.get("use_mel_loss", False):  # keep compatibility
        if "dataset_mode" not in config or config["dataset_mode"] == 'default' or config["dataset_mode"] == 'a2w' \
                or config["dataset_mode"] == 'a2w_mult' or config["dataset_mode"] == 'm2w' \
                or ('generator2_type' in config and config["dataset_mode"] == 'w2a'):
            if config.get("mel_loss_params", None) is None:
                criterion["mel"] = MelSpectrogramLoss(
                    fs=config["sampling_rate"],
                    fft_size=config["fft_size"],
                    hop_size=config["hop_size"],
                    win_length=config["win_length"],
                    window=config["window"],
                    num_mels=config["num_mels"],
                    fmin=config["fmin"],
                    fmax=config["fmax"],
                ).to(device)
            else:
                criterion["mel"] = MelSpectrogramLoss(
                    **config["mel_loss_params"],
                ).to(device)
        elif config["dataset_mode"] == 'art' or config["dataset_mode"] == 'a2m' or config["dataset_mode"] == 'w2a' or config["dataset_mode"] == 'm2a' \
                or config["dataset_mode"] == 'ph2a' or dataset_mode == 'ph2m':
            # note generator2_type + w2a still uses MelSpectrogramLoss
            criterion["mel"] = F.l1_loss
        else:
            raise ValueError("dataset_mode %s not supported" % config["dataset_mode"])
    else:
        config["use_mel_loss"] = False
    if config.get("use_inter_loss", False):  # keep compatibility
        pass
        # criterion["inter"] = InterLoss(
        #     **config["inter_loss_params"],
        # ).to(device)
    else:
        config["use_inter_loss"] = False
    if config["generator_params"].get("use_ph_loss", False):  # keep compatibility
        criterion["ph"] = F.cross_entropy
        config["use_ph_loss"] = True
    else:
        config["use_ph_loss"] = False

    # define special module for subband processing
    if config["generator_params"]["out_channels"] > 1 and config.get("pqmf", False):
        criterion["pqmf"] = PQMF(
            subbands=config["generator_params"]["out_channels"],
            # keep compatibility
            **config.get("pqmf_params", {}),
        ).to(device)

    # define optimizers and schedulers
    generator_optimizer_class = getattr(
        articulatory.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        articulatory.optimizers,
        # keep compatibility
        config.get("discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            logging.error('need to uncomment apex.parallel and DistributedDataParallel lines')
            exit()
            # from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "apex is not installed. please check https://github.com/NVIDIA/apex."
            )
        # model["generator"] = DistributedDataParallel(model["generator"])
        # model["discriminator"] = DistributedDataParallel(model["discriminator"])

    # show settings
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    logging.info(optimizer["generator"])
    logging.info(optimizer["discriminator"])
    logging.info(scheduler["generator"])
    logging.info(scheduler["discriminator"])
    for criterion_ in criterion.values():
        logging.info(criterion_)

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        checkpoint2_path = args.pretrain2 if len(args.pretrain2) != 0 else None
        trainer.load_checkpoint(args.pretrain, load_only_params=True, checkpoint2_path=checkpoint2_path)
        logging.info(f"Successfully loaded parameters from {args.pretrain}.")
        if len(args.pretrain2) != 0:
            logging.info(f"Successfully loaded parameters from {args.pretrain2}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
