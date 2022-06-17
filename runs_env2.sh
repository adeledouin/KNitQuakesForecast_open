#!/bin/bash

for j in 2_2_1 2_2_2 2_2_3

do
  python3 Main_KnitCity_RL.py --env 01 --results '_5023_newReward' --sub_set 'd_20_200' --model "$j" --gpu 0

  python3 Main_KnitCity_RL_transfert.py --env 01 --env_transfert 01 --results '_5023_newReward' --sub_set 'd_20_200' --model "$j" --gpu 0

  python3 Main_KnitCity_RL.py --env 03 --results '_5023_newReward' --sub_set 'd_20_200' --model "$j" --gpu 0

  python3 Main_KnitCity_RL_transfert.py --env 03 --env_transfert 03 --results '_5023_newReward' --sub_set 'd_20_200' --model "$j" --gpu 0

  python3 Main_KnitCity_RL.py --env 05 --results '_5023_newReward' --sub_set 'd_20_200' --model "$j" --gpu 0

	python3 Main_KnitCity_RL_transfert.py --env 05 --env_transfert 05 --results '_5023_newReward' --sub_set 'd_20_200' --model "$j" --gpu 0

done
