#!/bin/bash
declare -a methods=( "sm" "klinker" "relklinker" "stream" "predpath" )
declare -a datasets=( "datasets/synthetic/birthplace_deathplace.csv" "datasets/real/place_of_birth.csv" "datasets/synthetic/cross_Movies_vs_Directors.csv" "datasets/sample.csv" "datasets/real/derived_filtered_nationality_train_25facts.csv" "datasets/synthetic/cross_US_Presidents_vs_First_Lady.csv" "datasets/real/derived_filtered_profession_train_25facts.csv" "datasets/synthetic/predpath_civil_war_battle.csv" "datasets/real/education_degree.csv" "datasets/synthetic/predpath_company_president.csv" "datasets/real/institution.alm.csv" "datasets/sub_sample.csv" "datasets/synthetic/predpath_state_capital.csv" "datasets/sample.csv" "datasets/synthetic/Player_vs_Team_NBA.csv" "datasets/real/place_of_death.csv" "datasets/synthetic/predpath_vice_president.csv" )

for i in "${methods[@]}"
do
    for j in "${datasets[@]}"
    do
        python -m streamminer2 -d $j -o output -m $i
        sleep 60
    done
done
