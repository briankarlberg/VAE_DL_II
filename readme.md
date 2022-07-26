VAE_DL_II operations
Goal is to predict small-molecule drug response on
    cell-line expression data, coding and non-coding genes

# 2022-07-26
# sk learn make_regression test file
# 3000 samples, 250 features, 100 n_informative
# prescaled and split, next convert app to take numpy arrays
sbatch VAE-split.sh 25 v2 data/trn_mk_rg_v02.tsv data/val_mk_rg_v02.tsv data/tst_mk_rg_v02.tsv

# run transform and scaling functions externally, plus triple split
# devel notebook xfrm_v1.ipynb, in 7-25 devel dir
./VAE-split.sh 25 v0 data/train_set.csv data/val_set.csv data/test_set.val

# make minimal shell file from VAE-split.sh
# re-order main.py to take files in order of 1)train 2)test 3)val
# This is an 80-15-5 split

# mk_rg_02
./VAE-split_mk_rg.sh 25 mk_rg_02 data/3000x250_trn.tsv data/3000x250_tst.tsv data/3000x250_val.tsv

