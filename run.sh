#!/usr/bin/env bash

#conda activate python3.6
declare -x CUDA_VISIBLE_DEVICES=0 
declare -x CUDA_LAUNCH_BLOCKING=0 
python3 train.py --config config_v2.json --root_path "/rhome/eingerman/Neural/MachineCodes/Speech/ESPnet/egs2/blizzard2013/tts1/" --feature_stats_file "/rhome/eingerman/Neural/MachineCodes/Speech/ESPnet/egs2/blizzard2013/tts1/exp/tts_train_raw_phn_tacotron_g2p_en/decode_use_teacher_forcingtrue_train.loss.ave/stats/train/feats_stats.npz" --train_scp_file "/dump/raw/tr_no_dev/wav.scp" --valid_scp_file "/dump/raw/dev/wav.scp"