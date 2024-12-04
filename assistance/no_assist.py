from assistance.base_assistance import AssistanceMode
import numpy as np
from scipy.stats import differential_entropy
import ipdb


class NoAssist(AssistanceMode):

    def __init__(self, beta = None, sigmoid=0.001, alpha=1.0, saturate=True, num_actions=0, horizon=16, mins=None, maxs=None) -> None:
        self.beta = beta
        self.sigmoid = sigmoid
        self.alpha = alpha
        self.saturate = saturate
        self.mins = mins
        self.maxs = maxs
        self.horizon = horizon
        self.max_ent = np.log(np.prod(self.maxs-self.mins))
        self.min_ent = 0.5*np.log(self.beta**num_actions)+num_actions/2*(1.0+np.log(2*np.pi)) # human entropy
        super().__init__()
    
    def getPenalizedLikelihood(self, actions):
        # actions are num_rollouts x horizon x num_states
        return self.alpha * self.getLikelihood(actions)

    def getLikelihood(self,actions):
        return ((self.horizon*self.max_ent-self.getSumEntropy(actions)) / (self.horizon*(self.max_ent-self.min_ent)))**1.0
    
    def getSumEntropy(self,actions):
        sum_entropy = 0.0
        timesteps = np.shape(actions)[1] # horizon is second argument
        for tt in range(timesteps):
            sum_entropy += self.getEntropy(actions[:,tt,:])
        return sum_entropy

    def getEntropy(self,actions):      
        num_states = np.shape(actions)[-1]
        min_ent_per_state = self.min_ent / num_states
        with np.errstate(divide='ignore'):
            ent_per_state = differential_entropy(actions,method="ebrahimi")
            if self.saturate:
                for ii in range(len(ent_per_state)):
                    ent_per_state[ii] = np.maximum(ent_per_state[ii],min_ent_per_state)
            ent = np.sum(ent_per_state)

            return ent

    def getNumberOfModeParameters(self,actions):
        return 0.0 # nothing