#!/bin/bash

curl -v --compressed -F "images[]=https://img2.rrcimg.com/o_1blshai8q3509822328246865820739732.jpg?imageView/4/w/600/h/400" http://localhost:5000/classify 
curl -v --compressed -F "images[]=@images/o_1blp9bfc621457231430079257828839.jpg" http://localhost:5000/classify 
curl -v --compressed -F "images[]=https://img2.rrcimg.com/o_1blshai8q3509822328246865820739732.jpg?imageView/4/w/600/h/400" http://localhost:5001/localize
curl -v --compressed -F "images[]=@images/o_1blp9bfc621457231430079257828839.jpg" http://localhost:5001/localize
##curl --compressed -F "images[]=@images/o_1blp9bfc621457231430079257828839.jpg" http://localhost:5001/localize 2>/dev/null
##curl --compressed -F "images[]=@images/o_1bc4q9nfb1181222690274524715012536.jpg" -F "images[]=@images/o_1bf69iger15584463531465908144086.jpg"  http://localhost:5000/classify 2>/dev/null
##curl --compressed -F "images[]=@images/o_1bc4q9nfb1181222690274524715012536.jpg" -F "images[]=@images/o_1bf69iger15584463531465908144086.jpg"  http://localhost:5001/localize 2>/dev/null
