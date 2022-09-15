# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import enum
import logging
import os
import resampy

from multiprocessing import Manager

import numpy as np

from torch.utils.data import Dataset

from articulatory.utils import find_files
from articulatory.utils import read_hdf5


class AudioMelDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        mel_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(
            mel_files
        ), f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.mel_files = mel_files
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_utt_id:
            items = utt_id, audio, mel
        else:
            items = audio, mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class MelArtDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        mel_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
        transform=None,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query)) # includes path
        mel_files = sorted(find_files(root_dir, mel_query)) # includes path

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(
            mel_files
        ), f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.mel_files = mel_files
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]

        stage = root_dir.split('/')[1]
        feats_path = os.path.join('data', stage, 'feats.scp')
        assert os.path.exists(feats_path)
        with open(feats_path, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_lists = [l.split() for l in lines]
        fid_to_artp = {l_list[0]:l_list[1] for l_list in l_lists}
        art_files = []
        for fid in self.utt_ids:
            art_files.append(fid_to_artp[fid])
        self.art_files = art_files

        self.transform = ""
        if transform is not None:
            self.transform = transform

        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]

        # audio = self.audio_load_fn(self.audio_files[idx])

        mel = self.mel_load_fn(self.mel_files[idx]) # (T', C)
        art = np.load(self.art_files[idx]) # (T', C)
        mel = mel[:len(art), :]

        if self.transform == "10*f0":
            art[:,0] *= 10

        if self.return_utt_id:
            # items = utt_id, audio, mel, art
            items = utt_id, mel, art
        else:
            # items = audio, mel, art
            items = mel, art

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class SpeechDataset(Dataset):
    """PyTorch audio and articulatory feature dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        mel_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
        transform=None,
        input_transform=None,
        output_transform=None,
        spks=None,
        use_spk_id=False,
        use_ph=False,
        dataset_mode=None,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query)) # includes path
        mel_files = sorted(find_files(root_dir, mel_query)) # includes path

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        
        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(
            mel_files
        ), f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.mel_files = mel_files
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]

        stage = root_dir.split('/')[1]
        feats_path = os.path.join('data', stage, 'feats.scp')
        assert os.path.exists(feats_path)
        with open(feats_path, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_lists = [l.split() for l in lines]
        fid_to_artp = {l_list[0]:l_list[1] for l_list in l_lists}
        art_files = []
        for fid in self.utt_ids:
            art_files.append(fid_to_artp[fid])
        self.art_files = art_files

        spk2utt = None
        spk2utt_path = os.path.join('data', stage, 'spk2utt')
        if os.path.exists(spk2utt_path):
            with open(spk2utt_path, 'r') as inf:
                lines = inf.readlines()
            lines = [l.strip() for l in lines]
            l_lists = [l.split() for l in lines]
            spk2utt = {l[0]:l[1:] for l in l_lists}
        utt2spk = None
        utt2spk_path = os.path.join('data', stage, 'utt2spk')
        if os.path.exists(utt2spk_path):
            with open(utt2spk_path, 'r') as inf:
                lines = inf.readlines()
            lines = [l.strip() for l in lines]
            l_lists = [l.split() for l in lines]
            utt2spk = {l[0]:l[1] for l in l_lists}
        if spk2utt is None and utt2spk is not None:
            spk2utt = {}
            for utt in utt2spk:
                spk = utt2spk[utt]
                if spk in spk2utt:
                    spk2utt[spk].append(utt)
                else:
                    spk2utt[spk] = [utt]
        if utt2spk is None and spk2utt is not None:
            utt2spk = []
            for spk in spk2utt:
                utts = spk2utt[spk]
                for utt in utts:
                    utt2spk[utt] = spk
        if spks is None and spk2utt is not None:
            spks = sorted(list(spk2utt.keys()))
        if spks is not None:
            spks_set = set(spks)
            assert all([spk in spks_set for spk in spk2utt])
            spk2id = {spk:i for i, spk in enumerate(spks)}
        else:
            spk2id = None
        self.spks = spks
        self.spk2id = spk2id
        self.spk2utt = spk2utt
        self.utt2spk = utt2spk
        self.use_spk_id = use_spk_id
        if use_spk_id:
            assert utt2spk is not None and spk2id is not None
        self.use_ph = use_ph
        if self.use_ph:
            ph_path = os.path.join('data', stage, 'ph.scp')
            assert os.path.exists(ph_path)
            with open(ph_path, 'r') as inf:
                lines = inf.readlines()
            lines = [l.strip() for l in lines]
            l_lists = [l.split() for l in lines]
            fid_to_ph_p = {l_list[0]:l_list[1] for l_list in l_lists}
            ph_files = []
            for fid in self.utt_ids:
                ph_files.append(fid_to_ph_p[fid])
            self.ph_files = ph_files
        
        self.transform = transform
        self.input_transform = input_transform if input_transform is not None else transform
        self.output_transform = output_transform if output_transform is not None else transform

        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.audio_files))]

        if dataset_mode == 'ph2a' or dataset_mode == 'ph2m':
            self.use_audio = True  # NOTE should ideally be False but audio needed in collater
        else:
            self.use_audio = True
        if dataset_mode == 'ph2m' or dataset_mode == 'm2w':
            self.use_mel = True
        else:
            self.use_mel = False

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        utt_id = self.utt_ids[idx]
        art = np.load(self.art_files[idx]) # (T', C)
        if self.input_transform is not None:
            art = self.input_transform(art)
        items = {'art': art}
        if self.use_audio:
            audio = self.audio_load_fn(self.audio_files[idx])
            if self.output_transform is not None:
                audio = self.output_transform(audio)  # NOTE output_transform deprecated
            items['audio'] = audio
        if self.use_mel:
            mel = self.mel_load_fn(self.mel_files[idx]) # (T', C)
            mel = mel[:len(art), :]
            items['mel'] = mel
        if self.return_utt_id:
            items['utt_id'] = utt_id
        if self.use_spk_id:
            spk = self.utt2spk[utt_id]
            spk_id = self.spk2id[spk]
            items['spk_id'] = spk_id
        if self.use_ph:
            ph = np.load(self.ph_files[idx])
            items['ph'] = ph
        if self.allow_cache:
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class WavArtMultDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dirs,
        audio_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        audio_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
        transform=None,
        sampling_rate=None,
        sampling_rates=None,
        ignore_modalities=None,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        self.mod_is = []
        self.audio_files = []
        self.audio_load_fn = audio_load_fn
        self.utt_ids = []
        self.art_files = []
        ignore_modalities = set(ignore_modalities)
        for mod_i, root_dir in enumerate(root_dirs):
            if mod_i not in ignore_modalities:
                audio_files = sorted(find_files(root_dir, audio_query)) # includes path
                if audio_length_threshold is not None:
                    audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
                    idxs = [
                        idx
                        for idx in range(len(audio_files))
                        if audio_lengths[idx] > audio_length_threshold
                    ]
                    if len(audio_files) != len(idxs):
                        logging.warning(
                            f"Some files are filtered by audio length threshold "
                            f"({len(audio_files)} -> {len(idxs)})."
                        )
                    audio_files = [audio_files[idx] for idx in idxs]
                assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
                self.audio_files += audio_files
                if ".npy" in audio_query:
                    utt_ids = [os.path.basename(f).replace("-wave.npy", "") for f in audio_files]
                else:
                    utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
                self.utt_ids += utt_ids

                stage = root_dir.split('/')[1]
                feats_path = os.path.join('data', stage, 'feats.scp')
                assert os.path.exists(feats_path)
                with open(feats_path, 'r') as inf:
                    lines = inf.readlines()
                lines = [l.strip() for l in lines]
                l_lists = [l.split() for l in lines]
                fid_to_artp = {l_list[0]:l_list[1] for l_list in l_lists}
                art_files = []
                for fid in utt_ids:
                    art_files.append(fid_to_artp[fid])
                self.art_files += art_files

                self.mod_is += [mod_i]*len(audio_files)

        self.transform = ""
        if transform is not None:
            self.transform = transform

        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.audio_files))]

        self.sampling_rate = sampling_rate
        self.sampling_rates = sampling_rates

        assert len(self.audio_files) == len(self.art_files) and len(self.audio_files) == len(self.mod_is)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        modality_i = self.mod_is[idx]
        audio = resampy.resample(audio, self.sampling_rates[modality_i], self.sampling_rate)
        # mel = self.mel_load_fn(self.mel_files[idx]) # (T', C)
        art = np.load(self.art_files[idx]) # (T', C)

        if self.transform == "10*f0":
            art[:,0] *= 10

        if self.return_utt_id:
            # items = utt_id, audio, mel, art
            items = utt_id, audio, art, modality_i
        else:
            # items = audio, mel, art
            items = audio, art, modality_i

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class AudioDataset(Dataset):
    """PyTorch compatible audio dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*-wave.npy",
        audio_length_threshold=None,
        audio_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.waning(
                    f"some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.return_utt_id = return_utt_id
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio (T,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])

        if self.return_utt_id:
            items = utt_id, audio
        else:
            items = audio

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class MelDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dir,
        mel_query="*-feats.npy",
        mel_length_threshold=None,
        mel_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        if ".npy" in mel_query:
            self.utt_ids = [
                os.path.basename(f).replace("-feats.npy", "") for f in mel_files
            ]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(mel_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_utt_id:
            items = utt_id, mel
        else:
            items = mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)


class ArtDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dir,
        mel_query="*-feats.npy",
        mel_length_threshold=None,
        mel_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
        transform=None,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        if ".npy" in mel_query:
            self.utt_ids = [
                os.path.basename(f).replace("-feats.npy", "") for f in mel_files
            ]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        
        stage = root_dir.split('/')[1]
        feats_path = os.path.join('data', stage, 'feats.scp')
        assert os.path.exists(feats_path)
        with open(feats_path, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_lists = [l.split() for l in lines]
        fid_to_artp = {l_list[0]:l_list[1] for l_list in l_lists}
        art_files = []
        for fid in self.utt_ids:
            art_files.append(fid_to_artp[fid])
        self.art_files = art_files

        self.transform = ""
        if transform is not None:
            self.transform = transform
        
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(mel_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        # mel = self.mel_load_fn(self.mel_files[idx])
        art = np.load(self.art_files[idx])

        if self.transform == "10*f0":
            art[:,0] *= 10

        if self.return_utt_id:
            # items = utt_id, mel
            items = utt_id, art
        else:
            # items = mel
            items = art

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)
