#!/bin/bash

# Current directory 
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_ROOT=$( cd "$( dirname "${DIR}/../.." )" && pwd )

# Make modules available for import in Python
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Clean results
$DIR/clean.sh

# Generate datasets and get results for dataset-simple
python $DIR/generate_dataset_toy.py 2>&1 | tee -a generate_dataset_toy.log
python $DIR/scale_weight_effect_mlcsc_toy.py 2>&1 | tee -a scale_weight_effect_mlcsc_toy.log

# Generate datasets and get results for dataset-complex
python $DIR/generate_dataset.py 2>&1 | tee -a generate_dataset.log
python $DIR/scale_weight_effect_mlcsc.py 2>&1 | tee -a scale_weight_effect_mlcsc.log

echo 'done.'