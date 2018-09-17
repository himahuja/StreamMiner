#!/bin/bash
declare -a methods=( "sm" "klinker" "relklinker" "stream" "predpath" )
declare -a datasets=( "datasets/real/place_of_birth.csv" "datasets/sample.csv" "datasets/synthetic/Player_vs_Team_NBA.csv" "datasets/real/place_of_death.csv" )

for i in "${methods[@]}"
do
    for j in "${datasets[@]}"
    do
        python -m streamminer2 -d $j -o output -m $i
        sleep 60
    done
done
