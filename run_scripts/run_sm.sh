#!/bin/bash
cd ..
declare -a datasets=( "datasets/synthetic/cross_Movies_vs_Directors.csv" "datasets/real/derived_filtered_nationality_train_25facts.csv"  "datasets/real/derived_filtered_profession_train_25facts.csv" "datasets/real/place_of_death.csv" "datasets/real/place_of_birth.csv" "datasets/real/education_degree.csv" "datasets/real/institution.alm.csv" )

for i in "${datasets[@]}"
do
    python -m streamminer -d $i -o output -m sm
    sleep 60
done
