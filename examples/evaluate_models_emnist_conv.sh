#!/bin/bash

# RBM
echo 'Evaluate DBN-Conv-ProbMaxPool'
python3 evaluate_emnist.py --directory Eval_DBN_Conv_ProbMaxPool --file Model_DBN_Conv_ProbMaxPool.py --model Model_DBN_Conv_ProbMaxPool --tqdm
echo '\n\n\n'

#echo 'Evaluate DBN-Conv-ProbMaxProol'
#python3 evaluate_emnist.py --directory Eval_DBN --file Model_DBN.py --model Model_DBN --tqdm
#echo '\n\n\n'

