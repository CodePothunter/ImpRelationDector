#!/bin/bash
source /slfs1/users/xyw00/workspace/keras/bin/activate
THEANO_FLAGS=device=gpu,floatX=float32 python main.py

