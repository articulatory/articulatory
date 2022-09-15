# Deep Speech Synthesis from Articulatory Representations

> Pytorch implementation for deep articulatory synthesis.

Correspondence to: 

  - Peter Wu (peterw1@berkeley.edu)

## Paper

[**Deep Speech Synthesis from Articulatory Representations**](http://arxiv.org/abs/2209.06337)<br>
[Peter Wu](https://peter.onrender.com/), [Shinji Watanabe](https://sites.google.com/view/shinjiwatanabe), [Louis Goldstein](https://sail.usc.edu/~lgoldste/me/), [Alan W Black](https://www.cs.cmu.edu/~awb/), and [Gopala K. Anumanchipalli](https://www2.eecs.berkeley.edu/Faculty/Homepages/gopala.html)<br>
Interspeech 2022

If you find this repository useful, please cite our paper:

```
@inproceedings{peter2022artic,
  title={Deep Speech Synthesis from Articulatory Representations},
  author={Wu, Peter and Watanabe, Shinji and Goldstein, Louis and Black, Alan W and Anumanchipalli, Gopala Krishna},
  booktitle={Interspeech},
  year={2022}
}
```

## Installation

```bash
git clone https://github.com/articulatory/articulatory.git
cd articulatory
pip3 install -e .
```

## EMA-to-Speech

```bash
cd egs/ema/voc1
mkdir downloads
# download MNGU0 dataset and move emadata/ folder to downloads/
python3 local/mk_ema_feats.py
python3 local/pitch.py downloads/emadata/cin_us_mngu0 --hop 80
python3 local/combine_feats.py downloads/emadata/cin_us_mngu0 --feats pitch actions -o fnema
./run.sh --conf conf/e2w_hifigan.yaml --stage 1 --tag e2w_hifi --train_set mngu0_train_fnema --dev_set mngu0_val_fnema --eval_set mngu0_test_fnema
```

- Stage 1 in `./run.sh` is preprocessing and thus only needs to be run once per train-dev.-eval. triple. Stage 2 is training, so subsequent training experiments with the same data can use `./run.sh --stage 2`.
- Replace `conf/e2w_hifigan.yaml` with `conf/e2w_hifigan_car.yaml` to use our autoregressive model (HiFi-GAN CAR)

## Creating Your Own Speech Synthesizer

```bash
cd egs
mkdir <your_id>
cp -r TEMPLATE/voc1 <your_id>
```

- To use your own model, add the model code to a new file in `articulatory/models` and an extra line referencing that file in `articulatory/models/__init__.py`. Then, change `generator_type` or `discriminator_type` in the `.yaml` config to the name of the new model class.
- To customize the loss function, similarly modify the code in `articulatory/losses`. Then, call the loss function in `articulatory/bin/train.py`. Existing loss functions can be toggled on/off and modified through the `.yaml` config, e.g., in the "STFT LOSS SETTING" and "ADVERSARIAL LOSS SETTING" sections.

## Acknowledgements

Based on https://github.com/kan-bayashi/ParallelWaveGAN.
