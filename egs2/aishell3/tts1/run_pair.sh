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
if [ "${fs}" -eq 44100 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format wav "
fi

train_config=conf/train.yaml
inference_config=conf/decode.yaml

train_set=pair_no_dev_test_phn
valid_set=pair_dev_phn
test_sets=pair_test_phn
g2p=pypinyin_g2p_phone
# Input: 卡尔普陪外孙玩滑梯
# pypinyin_g2p: ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1
# pypinyin_g2p_phone: k a3 er3 p u3 p ei2 uai4 s un1 uan2 h ua2 t i1

# if you want to use officially provided phoneme text (better for the quality)
# train_set=train__no_dev_phn
# valid_set=dev_phn
# test_sets="dev_phn test_phn"
# g2p=none

./tts_pair.sh \
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
    ${opts} "$@" \
    --audio_format wav \
    --gpu_inference true \
    --teacher_dumpdir exp/tts_train_raw_phn_pypinyin_g2p_phone/decode_use_teacher_forcingtrue_valid.loss.best/ \
    --tts_stats_dir exp/stats_phn_with_pair \
    --vocoder_file "downloads/ljspeech_hifigan.v1/checkpoint-2500000steps.pkl"
