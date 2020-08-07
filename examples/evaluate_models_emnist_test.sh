
#!/bin/bash

# RBM
echo 'Evaluate RBM'
python3 evaluate_emnist.py --directory Check_RBM --file Model_RBM_CD.py --model Model_RBM_CD --tqdm --maxepochs 5 --testing
echo '\n\n\n'

echo 'Evaluate DBM'
python3 evaluate_emnist.py --directory Check_DBM --file Model_DBM_PCD.py --model Model_DBM_PCD --tqdm --maxepochs 5 --testing
echo '\n\n\n'

echo 'Evaluate DBN'
python3 evaluate_emnist.py --directory Check_DBN_Test_IntModule --file Model_DBN_IntModule.py --model Model_DBN_IntModule --tqdm --testing --maxepochs 5
echo '\n\n\n'

echo 'Evaluate DBN'
python3 evaluate_emnist.py --directory Check_DBN_Test --file Model_DBN.py --model Model_DBN --tqdm --testing --maxepochs 5
echo '\n\n\n'

echo 'Evaluate HW_RWS'
python3 evaluate_emnist.py --directory Check_HM_RWS --file Model_HM_RWS.py --model Model_HM_RWS --tqdm --testing --maxepochs 5
echo '\n\n\n'
