#!/bin/bash

# Current directory 
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Remove datasets
rm -rf ${DIR}/*.npz
rm -rf ${DIR}/*.pkl

# Remove experiment results
rm -rf ${DIR}/mlcsc-scale-weight-effect
rm -rf ${DIR}/mlcsc-scale-weight-effect-toy

# Remove figures
rm -rf ${DIR}/*.eps

# Remove logs
rm -rf ${DIR}/*.log
