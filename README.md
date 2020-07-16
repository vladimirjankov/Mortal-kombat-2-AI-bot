# Mortal kombat 2 Sega genesis AI bot

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
