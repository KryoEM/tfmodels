#!/bin/bash

export PYTHONPATH=/jasper/EMAN2/bin:/jasper/EMAN2/lib:/jasper/git/pyutils:/jasper/git/tfutils:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
