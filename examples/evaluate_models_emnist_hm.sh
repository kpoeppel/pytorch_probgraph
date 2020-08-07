#!/bin/bash

# RBM
#echo 'Evaluate RBM'
#python3 evaluate_emnist.py --directory Eval_RBM --file Model_RBM.py --model Model_RBM --tqdm
#echo '\n\n\n'

#echo 'Evaluate DBN'
#python3 evaluate_emnist.py --directory Eval_DBN --file Model_DBN.py --model Model_DBN --tqdm
#echo '\n\n\n'

#echo 'Evaluate DBM_PCD'
#python3 evaluate_emnist.py --directory Eval_DBM_PCD --file Model_DBM_PCD.py --model Model_DBM_PCD --tqdm
#echo '\n\n\n'

#echo 'Evaluate DBM_CD'
#python3 evaluate_emnist.py --directory Eval_DBM_CD --file Model_DBM_CD.py --model Model_DBM_CD --tqdm
#echo '\n\n\n'

echo 'Evaluate HW_WS'
python3 evaluate_emnist.py --directory Eval_HM_WS --file Model_HM_WS.py --model Model_HM_WS --tqdm
echo '\n\n\n'

echo 'Evaluate HM_RWS'
python3 evaluate_emnist.py --directory Eval_HM_RWS --file Model_HM_RWS.py --model Model_HM_RWS --tqdm
echo '\n\n\n'

