#!/bin/bash
cd ..
declare -a datasets=( "datasets/real/education_degree.csv" "datasets/real/institution.alm.csv" )

for i in "${datasets[@]}"
do
    python -m streamminer -d $i -o output -m sm
    sleep 60
done
