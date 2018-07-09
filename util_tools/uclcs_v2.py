#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted(
        [(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join(
        [('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    command = 'predicted_evidence=../fever/data/indexed_data/dev.sentences.p5.s5.ver20180629.jsonl '\
        'label_pred=/tmp/$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 20) '\
        'reader=esim_ir_pred_filtered_label_ver20180629 '\
        'bias1={bias1} '\
        'bias2={bias2} '\
        'bash jack_reader.sh'\
              .format(bias1=c['bias1'],
                        bias2=c['bias2'])
    return command


def to_logfile(c, path):
    outfile = "%s/uclcs_fb-ntn_v2.%s.log" % (path, summary(c).replace(
        "/", "_"))
    return outfile


def main(argv):
    hyperparameters_space = dict(
        bias1=[round(x * 0.10, 2) for x in range(-20, 20, 1)],
        bias2=[round(x * 0.10, 2) for x in range(-20, 20, 1)])
    configurations = list(cartesian_product(hyperparameters_space))

    path = './qsub_logs/fb-ntn/uclcs_v2'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/tyoneda/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

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

cd $HOME/jack

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 60 && {}'.format(
            job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
