# template

[![Code Coverage](https://codecov.io/gh/HephaestusProject/template/branch/master/graph/badge.svg)](https://codecov.io/gh/HephaestusProject/template)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

* abstract의 내용을 넣어주세요.

## Table

* 구현하는 paper에서 제시하는 benchmark dataset을 활용하여 구현하여, 논문에서 제시한 성능과 비교합니다.
  + benchmark dataset은 하나만 골라주세요.
    1. 논문에서 제시한 hyper-parameter와 architecture로 재현을 합니다.
    2. 만약 재현이 안된다면, 본인이 변경한 사항을 서술해주세요.

## Training history

* tensorboard 또는 weights & biases를 이용, 학습의 로그의 스크린샷을 올려주세요.

## OpenAPI로 Inference 하는 방법

* curl ~~~

## Usage

### Environment

* install from source code
* dockerfile 이용

### Training & Evaluate

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.
    - single-gpu, multi-gpu

* docker build
docker build . --file charlm-trainer.Dockerfile --tag charlm-trainer:v0.1 --rm

* docker run
docker run --interactive --tty --name clm --gpus all --shm-size 4G --volume /home/yongrae/pytorch-CharLM:/charlm charlm-trainer:v0.1

* Run
CUDA_VISIBLE_DEVICES=0 python main.py train --train-val-dir data/ptb --train-path train.txt --val-path valid.txt --word-vocabulary-path tokenizers/data/word_vocabulary.tsv --char-vocabulary-path tokenizers/data/char_vocabulary.tsv --max-word-length 65 --sequence-length 35 --char-embedding-dim 15 --char-conv-kernel-sizes '1,2,3,4,5,6' --char-conv-out-channels '25,50,75,100,125,150' --hidden-dim 300 --num-highway-layers 1 --use-batch-norm --dropout 0.5 --gradient-clip-val 5.0 --lr 1.0 --batch-size 20 --num-workers 4 --max-epochs 25


* Test
CUDA_VISIBLE_DEVICES=0 python main.py test --test-path data/ptb/test.txt --word-vocabulary-path tokenizers/data/word_vocabulary.tsv --char-vocabulary-path tokenizers/data/char_vocabulary.tsv --max-word-length 65 --sequence-length 35 --checkpoint-path results/runs/run/v071/checkpoints/epoch\=024_val_ppl\=81.84527.ckpt

### Inference

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.

### Project structure

* 터미널에서 tree커맨드 찍어서 붙이세요.

### License

* Licensed under an MIT license.
