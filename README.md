# RL Docking
let's see what we can do. 


- [ ] I am using stable base lines. 



## OpenMPI Modification 
We will try to build off stable-baselines as much as possible. For this project
we will NOT support recurrent policies.  


We will create two types of worker objects:
- Learners 
- Environments 

Learners have a single task, to train the network. Learners will collect data
from the environments and also provide actions. Environments will be simple and 
are given a step, return a state of the system, and reset the system. 

A baseline concurrency model would look like a bunch of severs 

1. t=1  Learner ---- sends action    --> environment
2. t=2  Learner <--- gets back state --  environment
3. t=3  Learner trains 

But there are a few issues with bottlenecks. Here, the learner task is constantly
having to manage communication with tons of environments. 

Instead, let us try to remove communication here by giving the environments 
their own models

1. t=1  Learner ---- weights --> environment + model
2. t=2  Because each environemnt has its own model, it can generate data with no synchronization and just pass it to the model. 
        The model after some period of collecting data updates weights on everyone. 
        
We will need to make sure that the network does not diverge. 
