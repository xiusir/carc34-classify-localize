# Localization model for car pictures
Method: localization as regression
## carc_class.py
Define text name of 34 classes.
## carc_flags.py
Define all configuable parameters.
## train.py
Run train process.
python train.py 
## evaluate.py
Run evaluate process and print predict results for every examples.
python evaluate.py 
## build_model.py
Build and export model named "frozen_custom.pb" to train_dir.
python build_model.py 
