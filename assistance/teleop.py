from assistance.base_assistance import AssistanceMode
import numpy as np
from scipy.stats import differential_entropy

class Teleoperation(AssistanceMode):

    def __init__(self, beta = None, sigmoid=0.001, alpha=0.97, saturate=True, num_actions=0, horizon=16, mins=None, maxs=None) -> None:
        self.beta = beta
        self.sigmoid = sigmoid
        self.alpha = alpha
        self.saturate = saturate # not used for this mechanism
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
        # can use analytical expression for multivariate gaussian entropy
        return self.min_ent

    def getNumberOfModeParameters(self, actions):
        timesteps = np.shape(actions)[1]
        num_states = np.shape(actions)[2]
        return timesteps*num_states # person would be responsible for the entire specification