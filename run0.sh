#!/bin/bash

#### run ML model 5023

#for j in v2_5_4 v1_2_1 v1_2_2 v1_2_3 v2_2_1 v2_2_2 v2_2_3 v3_2_1 v3_2_2 v3_2_3
#do
#  python3 /data/Douin/These/knit_quakes_forecast/fichier_to_transfert/KnitQuakesForecast/Main_remote_scalar.py -r -c '220524' "v22" "$j" '' 23 --train --cuda_device "cuda:0"
#done

#### map to reference set
for j in v2_5_4 v1_2_1 v1_2_2 v1_2_3 v2_2_1 v2_2_2 v2_2_3 v3_2_1 v3_2_2 v3_2_3
do
  python3 /data/Douin/These/knit_quakes_forecast/fichier_to_transfert/KnitQuakesForecast/Deal_data_remote_reference_sets.py -r '220524' "v22" "$j" 23
done

python3 /data/Douin/These/knit_quakes_forecast/fichier_to_transfert/KnitQuakesForecast/Deal_data_RL.py