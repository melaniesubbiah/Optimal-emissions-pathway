# Optimal-emissions-pathway
Learning an optimal policy for emissions reductions to meet global temperature goals using reinforcement learning.

You can read the writeup of this project in Emissions_planning.pdf

Sample command to train the model:

`python emissions_planning.py --name example_run  --seed 14 --timesteps 1000 --stdout`

The command line arguments are:
`name` - the name that will be given to the model output directory  
`action_space` - the max amount the model can increase or decrease emissions by, an integer  
`reward_mode` - which reward function to use (simple, temp, conc, carbon_cost, temp_emit, or temp_emit_diff)  
`forcing` - whether to use additional forcing factors in the FaIR simulator  
`output_path` - what directory to put the outputs in, typically 'outputs'  
`stdout` - whether to log to stdout or save the logging to a file  
`seed` - random seed to use  
`device` - what device to use for training  
`lr` - learning rate  
`n_steps` - number of steps to run for each environment per model update  
`gamma` - discount factor for the model  
`timesteps` - number of training timesteps   
`algorithm` - what RL algorithm to use (a2c, ppo, or ddpg)  
`multigas` - whether to use multigas mode for the FaIR simulator  
`scenario` - which RCP scenario to use  


