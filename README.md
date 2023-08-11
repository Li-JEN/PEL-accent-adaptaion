# Parameter-Efficient Learning for Text-to-Speech Accent Adaptation

The code of "Parameter-Efficient Learning for Text-to-Speech Accent Adaptation"

# Installation
- This repo depends on ESPnet2 with python 3.8
- Clone this repo and follow the installation of ESPnet: [Link](https://espnet.github.io/espnet/installation.html)

# Training
- The work modifies the recipe of JVS with the Parameter-efficient transfer learning (PETL).
```sh
$ cd egs2/jvs/tts1
```
- Please follow the tutorial of egs2/JVS and the standard fine-tuning can be done.

- To implement PETL:
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


# Citations
