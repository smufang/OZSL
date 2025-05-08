cd ..
python setup.py install
cd src

python train.py ../config/flo_seed.yaml 
python train_stage1_unseen_as_ood_seed.py --seed 5 --seed2 5 ../config/flo_der1_seed.yaml 
python test_split.py --seed 5 ../config/flo_seed.yaml 
python test_seed.py --seed 5 --seed2 5 ../config/flo_der1_seed.yaml 
