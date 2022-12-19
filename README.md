# FAS-Aug
 Bag of Augmentations for Generalized Face Anti-Spoofing
 
## Training
```bash
python train.py --trainer vit_convpass TRAIN.INIT_LR 1e-4 TRAIN.BETA 10.0 TRAIN.ALPHA 0.02 DATA.EXTRA_DOMAIN 3 TRAIN.PENALTY_MODE 'RnC' DATA.TRAIN_LIST "['data/data_list/CASIA-ALL.csv', 'data/data_list/MSU-MFSD-ALL.csv', 'data/data_list/REPLAY-ALL.csv']" DATA.TEST "data/data_list/OULU-NPU-ALL.csv"
```
Some important hyperparameters:
* `TRAIN.BETA`: The scaling factor for computing the SARE Loss. The default value is 10. 
* `TRAIN.ALPHA`: The scaling factor for computing the Supervised Constractive Loss. The default value is 0.02.
* `TRAIN.PENALTY_MODE`: The mode of loss penalty, either 'RnC', 'R' or 'C'. The default mode is 'RnC', which computes the empirical risk of spoofing examples and the Supervised Constrastive loss to real examples with the aid of `TRAIN.BETA` and `TRAIN.ALPHA`. If 'R'  is set, the training model only computes the empirical risk of all examples as pelnalty loss with the aid of `TRAIN.BETA`. Similarly, if 'C'  is set, the training model only computes the Supervised Constrastive loss to all examples as pelnalty loss with the aid of `TRAIN.ALPHA`.
* `TRAIN.INIT_LR`: The initial learning rate. We set it as 1e-4 when `--trainer vit_convpass` or `--trainer vit_adapter` and 1e-3 when `--trainer bc`
* `DATA.EXTRA_DOMAIN`: The number of additional domains for applying the FAS augmentation. The default value is 3.
