#!/bin/bash

make clean
make



./sd -nnbrs=20 -minsim=.3 sports.mat sports.nbrs
./sd_serial -nnbrs=20 -minsim=.3 sports.mat sports.nbrs 	 	
