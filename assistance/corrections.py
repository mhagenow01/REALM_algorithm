from assistance.base_assistance import AssistanceMode
import numpy as np
from scipy.stats import differential_entropy

class Corrections(AssistanceMode):

    def __init__(self,k, beta = None, sigmoid=0.001, alpha=0.98, saturate=True, num_actions=0, horizon=16, mins=None, maxs=None) -> None:
        self.k = k # correction dimension (dimension of latent space)
        self.corrections = None
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
        
        correction_dim = self.k

        #########################
        # first perform pca     #
        #########################

        # actions is num_rollouts x horizon x num_states
        num_rollouts = np.shape(actions)[0]
        timesteps = np.shape(actions)[1] # horizon
        num_states = np.shape(actions)[2]

        actions_latent_processed = np.zeros(np.shape(actions))
        rot_datas = np.zeros(np.shape(actions))
        mean_data = np.zeros((timesteps,num_states))

        # corrections will be num_corr_directions x horizon x num_states
        corrections = np.zeros(shape=(correction_dim,timesteps,num_states))

        prev_Vt = None
        flip = [1]*correction_dim

        for tt in range(timesteps):
            # PCA using SVD
            data_tmp = actions[:,tt,:] # num_rollouts x num_states
            mean_data[tt,:] = np.mean(data_tmp,axis=0)
            data_tmp_mr = data_tmp - mean_data[tt,:]
            U, s, Vt = np.linalg.svd(data_tmp_mr, full_matrices=False)
            
            # rotate data into principal component frame to make replacements
            rot_data = (Vt @ data_tmp_mr.T) # num_states x num_rollouts

            rot_datas[:,tt,:] = rot_data.T.copy()

            for ii in range(correction_dim):
                # replace high-variance data directions with samples from beta
                rot_data[ii,:] = np.random.normal(scale=np.sqrt(self.beta), size=num_rollouts)
                
                # store corrections -- note the flipping isn't really used here since this is recalculated per sample
                if prev_Vt is not None:
                    if np.dot(Vt[ii],prev_Vt[ii])<0.0:
                        flip[ii] = -1
                    else:
                        flip[ii] = 1
                
                corrections[ii,tt,:] = 3.0*flip[ii]*s[ii]*(1/np.sqrt(num_rollouts))*Vt[ii]
                
                if prev_Vt is None:
                    prev_Vt = Vt.copy()
                
            
                prev_Vt[ii] = flip[ii]*Vt[ii].copy()
        
            # rotate back, and store value with mean readded (post PCA)
            rot_back = Vt.T @ rot_data
            rot_back_ma = rot_back.T + np.mean(data_tmp,axis=0) # num_rollouts x num_states
            actions_latent_processed[:,tt,:] = rot_back_ma

        self.corrections = corrections.copy()
        self.actions_latent_processed = actions_latent_processed.copy()
        self.mean_data = mean_data.copy()
        self.rot_data = rot_datas.copy()

        sum_entropy = 0.0
        for tt in range(timesteps):
            sum_entropy += self.getEntropy(actions_latent_processed[:,tt,:])
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


    def getNumberOfModeParameters(self, actions):
        timesteps = np.shape(actions)[1]
        return self.k * timesteps # person would be responsible for differential input on 'k' axes over all timesteps
