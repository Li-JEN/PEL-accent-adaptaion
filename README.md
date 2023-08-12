# Parameter-Efficient Learning for Text-to-Speech Accent Adaptation

The code of "Parameter-Efficient Learning for Text-to-Speech Accent Adaptation"

## Installation
- This works depends on ESPnet2 with python 3.8
```sh
$ sudo apt-get install cmake
$ sudo apt-get install sox
$ sudo apt-get install zip
$ git clone https://github.com/Li-JEN/PEL-accent-adaptaion.git
$ cd PEL-accent-adaptaion/tools
$ ./setup_anaconda.sh miniconda espnet 3.8
$ ./activate_python.sh
$ make TH_VERSION=1.10.1 CUDA_VERSION=11.3
$ pip install pyopenjtalk
$ pip install typeguard==2.13.3
$ pip install Pillow==9.5.0
$ pip install numpy==1.23.0
```
- For more detailed installation, please refer to [official tutorial](https://espnet.github.io/espnet/installation.html)

## Training
- The work modifies the recipe of JVS with the Parameter-efficient transfer learning (PETL).
```sh
$ cd egs2/jvs/tts1
```
1. Please follow the [tutorial](https://github.com/espnet/espnet/blob/master/egs2/jvs/tts1/README.md) of egs2/jvs/tts1 to prepare required files. 
2. Because the Fastspeech2 need the duration alignment file for training duration predictor **You may need to run the adaptation with AR model (Tacotron) to bulid the teacher model for preparing duration** feature or checkout the newest ESPnet for integration with [MFA](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/tts1/README.md#1-data-preparation)
3. Before attempting PETL, please make sure you are able to implement vanilla fine-tuning with Fastspeech2 (Please don't skip first two steps, or there are missing files for training Fastspeech2)
```sh
 ./run.sh \
    --stage 5 \
    --g2p pyopenjtalk_accent_with_pause \
    --write_collected_feats true \
    --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    --train_config conf/tuning/finetune_fastspeech2.yaml \
    --train_args "--init_param downloads/0293a01e429a84a604304bf06f2cc0b0/exp/tts_train_fastspeech2_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth:tts:tts" \
    --tag finetune_fastspeech2_raw_phn_jaconv_pyopenjtalk_accent_with_pause
```
4. To implement PETL:
```sh
$ ./run.sh \
    --stage 5 \
    --g2p pyopenjtalk_accent_with_pause \
    --write_collected_feats true \
    --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    --train_config conf/tuning/finetune_fastspeech2_adapter.yaml \
    --train_args "--init_param downloads/0293a01e429a84a604304bf06f2cc0b0/exp/tts_train_fastspeech2_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth:tts:tts" \
    --freeze_param "tts.length_regulator tts.encoder.encoders tts.encoder.after_norm tts.decoder.after_norm tts.decoder.encoders tts.feat_out tts.postnet tts.gst tts.pitch_predictor tts.energy_predictor tts.duration_predictor tts.energy_embed tts.pitch_embed" \
    --tag finetune_fastspeech2_raw_phn_jaconv_pyopenjtalk_accent_with_pause_adapter
```
- We provide the configuration files of Adapter and Reprogramming under the folder egs2/jvs/conf/tuning


## Citations
```
@article{yang2023parameter,
  title={Parameter-Efficient Learning for Text-to-Speech Accent Adaptation},
  author={Yang, Li-Jen and Yang, Chao-Han Huck and Chien, Jen-Tzung},
  journal={arXiv preprint arXiv:2305.11320},
  year={2023}
}
```