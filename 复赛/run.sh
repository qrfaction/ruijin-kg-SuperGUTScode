#!/usr/bin/env bash

python prepocess.py
python prepocess_test.py

python train.py --fold 0123456789 --stage 1
python get_result.py --stage 1

python get_feature.py

python train.py --fold 0123456789 --stage 2
python get_result.py --stage 2

