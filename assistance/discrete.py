from assistance.base_assistance import AssistanceMode
import numpy as np
from scipy.stats import differential_entropy
from sklearn.cluster import KMeans

class Discrete(AssistanceMode):

    def __init__(self,k=2, beta = None, sigmoid=0.001, alpha=0.99, saturate=True, num_actions=0, horizon=16, mins=None, maxs=None) -> None:
        self.k = k
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
        # first divide trajectories into clusters
        num_rollouts = np.shape(actions)[0]
        actions_per_cluster = [ [] for _ in range(self.k) ] # list for each cluster

        flat_actions = actions.copy().reshape(actions.shape[0],-1) # num_rollouts x data (states over horizon)
        
        if (flat_actions == flat_actions[0]).all(): # checking rank takes too long -- instead look for noninuqe entries
            cluster_results = np.zeros((num_rollouts,),dtype=np.int8)
        else:
            clustering = KMeans(n_clusters = self.k, init='random', n_init=10).fit(flat_actions)
            cluster_results = clustering.fit_predict(flat_actions)

        rollouts_per_cluster = [] # how many trajectories end up in each cluster -- for weighted entropy average
        for cc in range(self.k):
            rollouts_per_cluster.append(np.sum(cluster_results==cc))

        
        
        for ii in range(num_rollouts):
            actions_per_cluster[cluster_results[ii]].append(actions[ii,:,:])

        # store to use in decision making
        self.rollouts_per_cluster = rollouts_per_cluster
        self.actions_per_clusters = actions_per_cluster

        sum_entropy = 0.0
        timesteps = np.shape(actions)[1] # horizon is second argument
        for cc in range(self.k): # number of clusters
            actions_per_cluster[cc] = np.array(actions_per_cluster[cc])
            for tt in range(timesteps):
                if rollouts_per_cluster[cc]>0:
                    sum_entropy += (float(rollouts_per_cluster[cc])/num_rollouts)*self.getEntropy(actions_per_cluster[cc][:,tt,:])
                    
                    
        return sum_entropy

    def getEntropy(self,actions):
        
        data_samples = np.shape(actions)[0] # data_samples x num_states

        # Practical consideration -- cannot calculate entropy if there aren't sufficient examples in the cluster...
        # this is more likely to occur when you overcluster (i.e., fitting 3 clusters when data is bimodal)

        num_states = np.shape(actions)[-1]

        if data_samples<5:
            return self.min_ent # teleoperation entropy -- assume a small amount of samples is akin to the sample coming from noisy people
        else:
            min_ent_per_state = self.min_ent / num_states
            with np.errstate(divide='ignore'):
                ent_per_state = differential_entropy(actions,method="ebrahimi")
                if self.saturate:
                    for ii in range(len(ent_per_state)):
                        ent_per_state[ii] = np.maximum(ent_per_state[ii],min_ent_per_state)
                ent = np.sum(ent_per_state)
           
            return ent

    def getNumberOfModeParameters(self, actions):
        return self.k # person would be responsible for choosing one of the 'k' modes