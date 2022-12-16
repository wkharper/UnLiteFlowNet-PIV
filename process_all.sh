#!/bin/bash
## Run this script from within the docker container to save ALL results automoatically

FLOWS="$(ls sample_data/)"

for FLOW in $FLOWS;
    do 
        echo $FLOW;
        python3 main.py --test --flow $FLOW --fps 15
        cp -r output/ SAVED_$FLOW/
        ./clean.sh
done 