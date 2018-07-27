#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging
import numpy as np


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted(
        [(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join(
        [('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    n_sentences2layers = {
        5: " ".join(np.interp(range(3), [0, 3], [5*4, 3]).astype(int).astype(str)),
        10: " ".join(np.interp(range(3), [0, 3], [10*4, 3]).astype(int).astype(str)),
        15: " ".join(np.interp(range(3), [0, 3], [15*4, 3]).astype(int).astype(str)),
        20: " ".join(np.interp(range(3), [0, 3], [20*4, 3]).astype(int).astype(str)),
        25: " ".join(np.interp(range(3), [0, 3], [25*4, 3]).astype(int).astype(str)),
        30: " ".join(np.interp(range(3), [0, 3], [30*4, 3]).astype(int).astype(str))
    }
    # 'reader=esim_ir_pred_filtered_label_ver20180629 '\
    command = 'python3 pipeline.py --config blah --model blah_mlp_s{0} --n_sentences_mlp {0} --layers {1}'\
              .format(c["n_sentences_mlp"], n_sentences2layers[c["n_sentences_mlp"]])
    return command


def to_logfile(c, path):
    outfile = "%s/.%s.log" % (path, summary(c).replace(
        "/", "_"))
    return outfile


def main(argv):
    hyperparameters_space = dict(
        n_sentences_mlp=[5, 10, 15, 20, 25, 30]
        )
    configurations = list(cartesian_product(hyperparameters_space))

    path = '/cluster/project2/mr/tyoneda/pipeline/fever/qsub_logs/fb-ntn/friday_sweep'

    # Check that we are on the UCLCS cluster first
    if not os.path.exists('/home/tyoneda/'):
        raise RuntimeError("/home/tyoneda not found")

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'MRR' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l tmem=16G
#$ -l h_rt=48:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /cluster/project2/mr/tyoneda/pipeline/fever

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 60 && {}'.format(
            job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
