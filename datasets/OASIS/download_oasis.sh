#!/bin/bash

for ((i=1; i<=12; i++))
do
   wget "https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc${i}.tar.gz"
   tar -zxvf "oasis_cross-sectional_disc${i}.tar.gz"
done
