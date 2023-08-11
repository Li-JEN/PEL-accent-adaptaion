#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format wav "
fi

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

# g2p=pypinyin_g2p
g2p=pypinyin_g2p_phone

# Input: 卡尔普陪外孙玩滑梯
# pypinyin_g2p: ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1
# pypinyin_g2p_phone: k a3 er3 p u3 p ei2 uai4 s un1 uan2 h ua2 t i1

# ./tts.sh \
#     --lang zh \
#     --feats_type raw \
#     --fs "${fs}" \
#     --n_fft "${n_fft}" \
#     --n_shift "${n_shift}" \
#     --win_length "${win_length}" \
#     --token_type phn \
#     --cleaner none \
#     --g2p "${g2p}" \
#     --train_config "${train_config}" \
#     --inference_config "${inference_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --srctexts "data/${train_set}/text" \
#     --use_xvector true \
#     --gpu_inference true \
#     --train_config conf/tuning/finetune_tacotron2.yaml \
#     --train_args "--init_param /home/espnet/egs2/aishell3/tts1/exp/tts_train_raw_phn_pypinyin_g2p_phone/train.loss.best.pth:tts:tts" \
#     --tag finetune_tacotron2_raw_phn_indomain_accent_lowlr \
#     ${opts} "$@"

./tts.sh \
    --lang zh \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_xvector true \
    --gpu_inference true \
    --write_collected_feats true \
    --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/decode_use_teacher_forcingtrue_train.loss.ave/ \
    --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/stats_phn \
    --train_config conf/tuning/finetune_fastspeech2.yaml  \
    --train_args "--init_param ../../aishell3/tts1/exp/tts_train_gst+xvector_conformer_fastspeech2_raw_phn_pypinyin_g2p_phone/train.loss.ave_5best.pth:tts:tts" \
    --tag finetune_fastpeech2_hlr \
    --inference_config conf/tuning/decode_fastspeech.yaml \
    ${opts} "$@" 


./tts.sh \
    --lang zh \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_xvector true \
    --gpu_inference true \
    --write_collected_feats true \
    --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/decode_use_teacher_forcingtrue_train.loss.ave/ \
    --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/stats_phn \
    --train_config conf/tuning/finetune_fastspeech2.yaml  \
    --train_args "--init_param ../../aishell3/tts1/exp/tts_train_gst+xvector_conformer_fastspeech2_raw_phn_pypinyin_g2p_phone/train.loss.ave_5best.pth:tts:tts" \
    --tag finetune_fastpeech2_decoder_hlr \
    --inference_config conf/tuning/decode_fastspeech.yaml \
     --freeze_param "tts.length_regulator tts.encoder tts.encoder.after_norm tts.feat_out tts.postnet tts.gst tts.pitch_predictor tts.energy_predictor tts.duration_predictor tts.energy_embed tts.pitch_embed" \
    ${opts} "$@" 



    # --freeze_param "tts.length_regulator tts.encoder tts.encoder.after_norm tts.feat_out tts.postnet tts.gst tts.pitch_predictor tts.energy_predictor tts.duration_predictor tts.energy_embed tts.pitch_embed" \
# # ADAPTER
#     ./tts.sh \
#     --lang zh \
#     --feats_type raw \
#     --fs "${fs}" \
#     --n_fft "${n_fft}" \
#     --n_shift "${n_shift}" \
#     --win_length "${win_length}" \
#     --token_type phn \
#     --cleaner none \
#     --g2p "${g2p}" \
#     --train_config "${train_config}" \
#     --inference_config "${inference_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --srctexts "data/${train_set}/text" \
#     --use_xvector true \
#     --gpu_inference true \
#     --write_collected_feats true \
#     --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/decode_use_teacher_forcingtrue_train.loss.ave/ \
#     --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/stats_phn \
#     --train_config conf/tuning/finetune_fastspeech2_adapter.yaml  \
#     --train_args "--init_param ../../aishell3/tts1/exp/tts_train_gst+xvector_conformer_fastspeech2_raw_phn_pypinyin_g2p_phone/train.loss.ave_5best.pth:tts:tts" \
#     --tag finetune_fastpeech2_adapter_hlr \
#     --inference_config conf/tuning/decode_fastspeech.yaml \
#      --freeze_param "tts.length_regulator tts.encoder.encoders tts.encoder.after_norm tts.decoder.after_norm tts.decoder.encoders tts.feat_out tts.postnet tts.gst tts.pitch_predictor tts.energy_predictor tts.duration_predictor tts.energy_embed tts.pitch_embed" \
#     ${opts} "$@" 
# # REprogramming ALL
# ./tts.sh \
#     --lang zh \
#     --feats_type raw \
#     --fs "${fs}" \
#     --n_fft "${n_fft}" \
#     --n_shift "${n_shift}" \
#     --win_length "${win_length}" \
#     --token_type phn \
#     --cleaner none \
#     --g2p "${g2p}" \
#     --train_config "${train_config}" \
#     --inference_config "${inference_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --srctexts "data/${train_set}/text" \
#     --use_xvector true \
#     --gpu_inference true \
#     --write_collected_feats true \
#     --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/decode_use_teacher_forcingtrue_train.loss.ave/ \
#     --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/stats_phn \
#     --train_config conf/tuning/finetune_fastspeech2_reprogram.yaml  \
#     --train_args "--init_param ../../aishell3/tts1/exp/tts_train_gst+xvector_conformer_fastspeech2_raw_phn_pypinyin_g2p_phone/train.loss.ave_5best.pth:tts:tts" \
#     --tag finetune_fastpeech2_reprogram_hlr \
#     --inference_config conf/tuning/decode_fastspeech.yaml \
#      --freeze_param "tts.length_regulator tts.encoder.encoders tts.encoder.after_norm tts.decoder.after_norm tts.decoder.encoders tts.feat_out tts.postnet tts.gst tts.pitch_predictor tts.energy_predictor tts.duration_predictor tts.energy_embed tts.pitch_embed" \
#     ${opts} "$@" 


# ## REprogramming first
# ./tts.sh \
#     --lang zh \
#     --feats_type raw \
#     --fs "${fs}" \
#     --n_fft "${n_fft}" \
#     --n_shift "${n_shift}" \
#     --win_length "${win_length}" \
#     --token_type phn \
#     --cleaner none \
#     --g2p "${g2p}" \
#     --train_config "${train_config}" \
#     --inference_config "${inference_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --srctexts "data/${train_set}/text" \
#     --use_xvector true \
#     --gpu_inference true \
#     --write_collected_feats true \
#     --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/decode_use_teacher_forcingtrue_train.loss.ave/ \
#     --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_indomain_accent_lowlr/stats_phn \
#     --train_config conf/tuning/finetune_fastspeech2_reprogram_input.yaml  \
#     --train_args "--init_param ../../aishell3/tts1/exp/tts_train_gst+xvector_conformer_fastspeech2_raw_phn_pypinyin_g2p_phone/train.loss.ave_5best.pth:tts:tts" \
#     --tag finetune_fastpeech2_reprogram_input_hlr \
#     --inference_config conf/tuning/decode_fastspeech.yaml \
#     --freeze_param "tts.length_regulator tts.encoder.encoders tts.encoder.after_norm tts.decoder.after_norm tts.decoder.encoders tts.feat_out tts.postnet tts.gst tts.pitch_predictor tts.energy_predictor tts.duration_predictor tts.energy_embed tts.pitch_embed" \
#     ${opts} "$@" 