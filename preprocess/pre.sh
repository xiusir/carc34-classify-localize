
cut -d' ' -f1,2 label_car_images.list | cut -d' ' -f2 | sort | uniq -c > label.stat
awk 'NR==FNR{if($1>100){s[$2]=1}}NR!=FNR&&($2 in s){print}' label.stat label_car_images.list | 
  awk 'BEGIN{cnt=0}{if(!($2 in s)){s[$2]=cnt;cnt=cnt+1}; print s[$2]" "$0}' |
  awk '{
    label=$1;
    path=$2;
    n=split(path,p,"/");
    bucket="image/"label"/"p[2]"/";
    name=p[5]
    print bucket,name,$0
  }' > label.dat 

while read newpath name label file others ; do 
  mkdir -p $newpath
  cp $file $newpath
done < label.dat
