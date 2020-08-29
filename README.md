# Mortal kombat 2 Sega genesis AI bot
Implementation of Mortal Kombat AI bot based on reinforcment learning.   
The neural network arhitecture consist of:  
Conv Layer with 128 filter with kernel size 3 and input shape 4, 160 112  
Conv Layer with 64 filter with kernel size 4  
Global average pooling 2D  
Dense layer 256  
Dense layer 128  
Dense with number of actions  

## Req
Python 3.6 requirements:
-Gym-retro  
-Gym  
-Tensorflow 1.15  
-Keras 1.15  
-KerasRL (with some modifications)  
-Baselines  
-wandb  

## Files  

scripts:   
	mortalkombat_env.py - setup of retro env for mortal kombat.   
	Implementaion of custom action and training wrapper for the env.
	train_model.py - training of deep q neural network  
	test_model.py - test for deep q neural network   
	train.sh - runes the the training script on slurm workload manager  
	test.sh - runes the the test script on slurm workload manager  
  
models:  
	Trained models for Liu Kang.   
	14 models present  
