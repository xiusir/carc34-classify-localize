#!/bin/bash

#python carc19_eval.py > eval.log

grep "/cn/" eval.log | cut -d'/' -f10-  | sed "s/['/]/ /g" | sed "s/\[//g;s/]//g" | 
  awk '
  function abs(num)
  {
    return (num > 0) ? num : (-num);
  }
  {
    diff=abs($5-$10)+abs($6-$11)+abs($7-$12)+abs($8-$13) ;
    printf "%s %s %s %s %s %s %s |* %.4lf *| %s %s %s %s\n", $3,$3,$1,$10,$11,$12,$13,diff,$5,$6,$7,$8
  }' | awk 'NR==FNR{s[$2]=$3" "$4" "$5" "$6}NR!=FNR{print $0" | "s[$1]}' ../tmp/carc34/boxpos_for_test.dat - |
   tee eval.csv | sort -nrk9,9 > ../tmp/carc34/eval.csv
  #awk '!($2 in s){print; s[$2]=1}' | 
#grep "/cn/" eval.log | cut -d'/' -f10-  | sed "s/['/]/ /g" | sed "s/\[//g;s/]//g" | awk '{printf "image/%s/cn/ %s %s %s %s %s\n", $1,$3,$10,$11,$12,$13}' | awk '!($2 in s){print; s[$2]=1}' > boxpos_for_train.dat
