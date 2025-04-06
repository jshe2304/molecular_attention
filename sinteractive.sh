#!/bin/sh

if [[ "$1" == "cpu" ]]; then

	sinteractive \
		--account=pi-andrewferguson \
		--partition=gm4-pmext \
		--time=04:00:00 \
		--nodes=1 \
		--ntasks-per-node=8 \
		--mem-per-cpu=2000

elif [[ "$1" == "gpu" ]]; then

	sinteractive \
		--account=pi-andrewferguson \
		--partition=andrewferguson-gpu \
		--time=04:00:00 \
		--gres=gpu:1 \
		--nodes=1 \
		--ntasks-per-node=8 \

else

	echo "Unknown argument."

fi
