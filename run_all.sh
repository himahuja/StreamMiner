#!/bin/bash
python -m streamminer -d datasets/sample.csv -o output -m klinker
sleep 5m
python -m streamminer2 -d datasets/sample.csv -o output -m sm
sleep 5m
python -m streamminer -d datasets/sample.csv -o output -m stream
sleep 5m
python -m streamminer -d datasets/sample.csv -o output -m predpath
sleep 5m
python -m streamminer -d datasets/sample.csv -o output -m relklinker
sleep 5m
python -m streamminer -d datasets/real/place_of_birth.csv -o output -m stream
sleep 5m
python -m streamminer -d datasets/real/place_of_birth.csv -o output -m klinker
sleep 5m
python -m streamminer -d datasets/real/place_of_birth.csv -o output -m predpath
sleep 5m
python -m streamminer -d datasets/real/place_of_birth.csv -o output -m relklinker
sleep 5m
python -m streamminer2 -d datasets/real/place_of_birth.csv -o output -m sm
sleep 5m
python -m streamminer -d datasets/synthetic/Player_vs_Team_NBA.csv -o output -m stream
sleep 5m
python -m streamminer -d datasets/synthetic/Player_vs_Team_NBA.csv -o output -m klinker
sleep 5m
python -m streamminer -d datasets/synthetic/Player_vs_Team_NBA.csv -o output -m predpath
sleep 5m
python -m streamminer -d datasets/synthetic/Player_vs_Team_NBA.csv -o output -m relklinker
sleep 5m
python -m streamminer2 -d datasets/synthetic/Player_vs_Team_NBA.csv -o output -m sm
