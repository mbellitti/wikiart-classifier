#! /bin/bash

while read p; do
    wget -c -O $p
done <img_urls.txt

