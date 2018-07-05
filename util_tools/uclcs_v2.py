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
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    command = 'PYTHONPATH=. anaconda-python3-gpu ./bin/ntp2-cli.py ' \
              '--train data/ntp2/fb-ntn/train.tsv ' \
              '--dev data/ntp2/fb-ntn/dev.tsv ' \
              '--test data/ntp2/fb-ntn/test.tsv ' \
              '-c data/ntp2/fb-ntn/clauses.pl ' \
              '-E ntn -e 100 --max-depth 1 -b {} ' \
              '--corrupted-pairs {} --l2 {} --k-max 10 --all ' \
              '-F {} -R {} -I {} --learning-rate {} ' \
              '--nms-m {} --nms-efc {} --nms-efs {} ' \
              '--seed {} --decode' \
              ''.format(c['b'],
                        c['corrupted_pairs'],
                        c['l2'],

                        c['F'],
                        c['R'],
                        c['I'],
                        c['lr'],

                        c['nms_m'],
                        c['nms_efc'],
                        c['nms_efs'],

                        c['seed'])
    return command


def to_logfile(c, path):
    outfile = "%s/uclcs_fb-ntn_v2.%s.log" % (path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyperparameters_space = dict(
        seed=[0],
        corrupted_pairs=[1],
        l2=[0.001],
        b=[1000, 5000, 10000, 50000],

        F=[1, 2, 5, 10],
        R=[1, 2, 5, 10],
        I=[100],
        lr=[0.001, 0.005, 0.01, 0.05, 0.1],

        nms_m=[15],
        nms_efc=[100],
        nms_efs=[100]
    )
    configurations = list(cartesian_product(hyperparameters_space))

    path = './logs/fb-ntn/uclcs_v2'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
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

cd $HOME/workspace/ntp

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 60 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
