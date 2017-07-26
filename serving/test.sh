#!/bin/bash

time curl --compressed -v -F "images[]=@images/o_1bc4q9nfb1181222690274524715012536.jpg" -F "images[]=@images/o_1bf69iger15584463531465908144086.jpg"  http://localhost:5000/classify
time curl --compressed -v -F "images[]=@images/o_1bc4q9nfb1181222690274524715012536.jpg" -F "images[]=@images/o_1bf69iger15584463531465908144086.jpg"  http://localhost:5001/localize
