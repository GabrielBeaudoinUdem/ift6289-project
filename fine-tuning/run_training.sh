#!/bin/bash
python3 save_base_models.py

python3 train_mistral.py --task wiki
python3 train_mistral.py --task fl

python3 train_qwen.py --task wiki
python3 train_qwen.py --task fl


