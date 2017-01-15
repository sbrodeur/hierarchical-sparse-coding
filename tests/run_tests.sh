#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export DATASET_PATH=${DIR}/data

# Create output directories
mkdir -p ${DIR}/report/profile
mkdir -p ${DIR}/report/coverage

# Cleanup previous reports
target=${DIR}/report/coverage/
if find "$target" -mindepth 1 -print -quit | grep -q .; then
    # Output folder not empty, erase all existing files
    rm ${DIR}/report/coverage/*
fi

# Run unit tests and coverage analysis for the 'action' module
nosetests --verbosity 3 --with-coverage --cover-html --cover-html-dir=${DIR}/report/coverage \
--cover-erase --cover-tests --cover-package=hsc,hsc.dataset,hsc.analysis,hsc.utils,hsc.modeling
if type "x-www-browser" > /dev/null; then
	x-www-browser ${DIR}/report/coverage/index.html
fi

# For profiler sorting options, see:
# https://docs.python.org/2/library/profile.html#pstats.Stats

#PROFILE_FILE=${DIR}/report/profile/profile.out
#python -m cProfile -o ${PROFILE_FILE} `which nosetests` ${DIR}
#runsnake ${PROFILE_FILE}
