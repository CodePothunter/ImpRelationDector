#!/bin/bash

qsub -cwd -o log_filter500 -e error -S /bin/bash -P cpu.p  run.sh
