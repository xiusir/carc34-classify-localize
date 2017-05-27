# CARC19
Image classifier based on cifar10, the sample model from the official tutorial of tensorflow. 
Images about a car outside or inside and from different angles with 19 target classes.

# Benchmark
There are two datasets in project.
Every example contains a image sized 256x256 and a text label.

## big34w
 precision @ 1 = 0.994 ~ 0.997
### train - label_for_train.dat
 about 300,000 case
### test - label_for_test.dat
 about 40,000 case

## small3k
### train - label_for_train.dat
 about 2,000 case
### test - label_for_test.dat
 about 1,000 case

# HowToRun
* Install tensorflow with gpu support
* Clone: git clone carc19....
* SetWorkDir: 
    TFWORKDIR=/home/xxx/carc19_work/tmp  # your work dir
    mkdir -p $TFWORKDIR 
* ExtractData: 
    cd carc19/dataset/small3k 
    tar zxf image.tar.gz ;    # about 800MB
    mv carc19/dataset/small3k $TFWORKDIR/carc19
* Change:
    cd carc19/model
    modify carc19.py: tf.app.flags.DEFINE_string('tf_home', '/home/xxx/carc19_work/tmp', ... 
* Train:
    python carc19_train.py   # run about 2hours
* Evaluate:
    python carc19_eval.py    # evaluate for precision

# Reference:
  https://www.tensorflow.org/tutorials/deep_cnn

My email: xiusir#qq.com
