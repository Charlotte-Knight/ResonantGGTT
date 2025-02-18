#!/usr/bin/env bash

cd %(PWD)s
source setup.sh
export LOW_MASS_MODE=%(LOW_MASS_MODE)s
python %(COMMAND)s