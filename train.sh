#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python hugging/deberta.py
