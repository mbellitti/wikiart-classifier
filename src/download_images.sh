#! /bin/bash

while read p; do
    wget -c -O $p
done <missing_img_urls.txt
