#/bin/bash


grep prediction eval.out | cut -d' ' -f2,5 | sed "s/([^:]\+//g;s/)//g" | sort | uniq -c
