#!/bin/bash


function label_dict() {
  find train/ -name *.jpg | 
    cut -d'/' -f4- | 
    sed "s/o_[a-z0-9]\+.jpg//g" | 
    sort | uniq -c | awk '$1>10' | 
    grep -v "三方平台" | 
    awk '{print (NR-1)" "$2}' > label.dict
}

 
#image/6/cd/ o_1bd8ln90s4915845503649113290644673.jpg 6 image/cd/car/011912b7f3f44eb4/o_1bd8ln90s4915845503649113290644673.jpg 轮胎 img2.rrcimg.com/o_1bd8ln90s4915845503649113290644673.jpg?imageView/4/w/600/h/400 6
function generate_image() {
  find train/ -name *.jpg | 
    awk '{
      match($0, "train/.+o_"); 
      class=substr($0, RSTART, RLENGTH-2); 
      n=split(class,t,"/"); 
      class=t[4]"/"t[5]; 
      carid=t[3]; 
      match($0, "o_[a-z0-9]+.jpg"); 
      fname=substr($0, RSTART, RLENGTH); 
      print fname,class,$0,carid;
      }' | 
    awk '
      NR==FNR{c[$2]=$1}
      NR!=FNR&&($2 in c){
        printf "image/%s/cn/ %s %s %s %s %s\n", c[$2], $1, c[$2], $3, $2, $4
      } ' label.dict - |
      tee label_all.dat |
      while read path fname class fullpath others; do
        mkdir -p $path
        cp $fullpath $path/$fname
      done

  python image_cutter.py -i image &>cutter.log 
  grep skip cutter.log | cut -d'/' -f4 | awk 'NR==FNR{s[$1]=1}NR!=FNR&&!($2 in s){print}' - label_all.dat > tmp.dat && mv tmp.dat label_all.dat
}


function split_train_test() {
  cut -d' ' -f6 label_all.dat | sort -u | awk 'NR%5==3' > test_case
  awk 'NR==FNR{s[$1]=1}NR!=FNR&&($NF in s){print}' test_case label_all.dat > label_for_test.dat
  awk 'NR==FNR{s[$1]=1}NR!=FNR&&!($NF in s){print}' test_case label_all.dat > label_for_train.dat
}

generate_image

