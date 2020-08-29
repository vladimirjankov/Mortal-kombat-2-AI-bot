# Mortal kombat 2 Sega genesis AI bot
Implementation of Mortal Kombat AI bot based on reinforcment learning. 
Deep Q-learning model is trained with gym-retro python package and tensorflow.
## The neural network arhitecture    
Conv Layer with 128 filter with kernel size 3 and input shape 4, 160 112  
Conv Layer with 64 filter with kernel size 4  
Global average pooling 2D  
Dense layer 256  
Dense layer 128  
Dense with number of actions  
![alt text](https://i.ytimg.com/vi/-gl71qZoZw8/hqdefault.jpg)

## Requirements
Python 3.6 requirements:
-Gym-retro  
-Gym  
-Tensorflow 1.15  
-Keras 1.15  
-KerasRL (with some modifications)  
-Baselines  
-wandb  

## Files  

### Scripts     
mortalkombat_env.py - setup of retro env for mortal kombat.   
Implementaion of custom action and training wrapper for the env.
train_model.py - training of deep q neural network  
test_model.py - test for deep q neural network   
train.sh - runes the the training script on slurm workload manager  
test.sh - runes the the test script on slurm workload manager  
  
### Models  
Trained models for Liu Kang.   
14 models present  
