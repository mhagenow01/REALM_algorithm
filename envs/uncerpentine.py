import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
project_path = dir_path+'/..'

sys.path.append(project_path+'/policies')
sys.path.append(project_path)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bezier
import pickle


from tqdm import tqdm
from assistance.no_assist import NoAssist
from assistance.teleop import Teleoperation
from assistance.discrete import Discrete
from assistance.corrections import Corrections
from policies.diffusion import Diffusion

"""
Environment for testing multi-mechanism assistance (simulated study)
"""
class Uncerpentine():

    def __init__(self) -> None:
        self.samples_per_bezier_pt = 50
        self.noise = 0.001
        self.margin = 0.1

    
    def getStatesActions(self,samples,train_samples):
        states = []
        actions = []
        num_train = len(train_samples)

        for ll in range(num_train):    
            states_tmp = samples[train_samples[ll]].copy()
            # absolute actions
            actions_tmp = states_tmp[:,1:].copy()
            actions_tmp = np.concatenate((actions_tmp,actions_tmp[:,-1].reshape((2,1))),axis=1)
            states.append(states_tmp)
            actions.append(actions_tmp)
        return states, actions

    def generateEnv(self, length=500, uncertainty_profiles=[], starts=[], ends=[], num_samples=300):
        # Generate base behavior using bezier curves
        num_nodes = 4
        x1 = self.margin+0.1*np.random.rand()
        y1 = self.margin+0.1*np.random.rand()

        x2 = 0.3+0.1*np.random.rand()
        y2 = 1.0 - self.margin - 0.3*np.random.rand()

        x3 = 0.6+0.1*np.random.rand()
        y3 = self.margin+0.3*np.random.rand()

        x4 = 1.0-self.margin-0.1*np.random.rand()
        y4 = 1.0 - self.margin - 0.3*np.random.rand()

        xs = np.array([x1,x2,x3,x4])
        ys = np.array([y1,y2,y3,y4])

        nodes = np.stack([xs,ys],axis=0) # 2 x num_nodes
        bz = bezier.Curve(nodes, degree=num_nodes-1)

        mean_traj = bz.evaluate_multi(np.linspace(0.0, 1.0, length)) # 2 x num_samples

        mean_traj = np.hstack((mean_traj,np.tile(mean_traj[:,[-1]],100)))
    
        tangent_traj = []
        tts = np.linspace(0.0, 1.0, length)
        for ll in range(length-1):
            diff = mean_traj[:,ll+1]-mean_traj[:,ll]
            diff = np.array([-diff[1],diff[0]]) # tangent to normal
            tangent_traj.append(diff / np.linalg.norm(diff)) # save unit vector
            if ll==length-2:
                tangent_traj.append(diff / np.linalg.norm(diff)) # repeat last vector
        
        tangent_traj = np.array(tangent_traj).squeeze().T # 2 x num_samples

        # Generate samples
        samples = []

        for ss in range(num_samples):
            samp_tmp = mean_traj.copy()
            # add noise
            for tt in range(np.shape(samp_tmp)[1]):
                samp_tmp[:,tt]+=np.random.normal(loc=0.0,scale=self.noise,size=np.shape(samp_tmp)[0])
            
            # + np.random.multivariate_normal(0*np.,self.noise,*np.shape(mean_traj))
            for ii, prof in enumerate(uncertainty_profiles):
                start_tmp = starts[ii]
                end_tmp = ends[ii]
                range_tmp = end_tmp-start_tmp

                if prof=='discrete':
                    coin_flip = np.sign(np.random.rand()-0.5)
                    if ss==0:
                        amp = np.random.rand()*0.05+0.02
                        dir_tmp = tangent_traj[:,start_tmp]
                    samp_tmp[0,start_tmp:end_tmp] += dir_tmp[0]*coin_flip*amp*np.sin(np.pi*np.linspace(0,1,range_tmp))
                    samp_tmp[1,start_tmp:end_tmp] += dir_tmp[1]*coin_flip*amp*np.sin(np.pi*np.linspace(0,1,range_tmp))
                if prof=='correction':
                    # add some uncertainty just in the velocity direction
                    coin_flip = np.sign(np.random.rand()-0.5)
                    phase = np.random.rand()*0.5+0.2 
                    sub_freq = np.random.rand()*4+1.5
                    sin_mod = coin_flip * 0.05 * np.random.rand()*np.sin(np.pi*np.linspace(0,1,range_tmp))**phase*np.sin(sub_freq*np.pi*np.linspace(0,1,range_tmp))
                    # sin_mod = 0.1*np.ones((range_tmp))
                    for tt in range(start_tmp,end_tmp):
                        samp_tmp[:,tt]+= tangent_traj[:,tt]*sin_mod[tt-start_tmp]
                if prof=='teleoperation':
                    start_pt = mean_traj[:,start_tmp]
                    end_pt = mean_traj[:,end_tmp]

                    a = np.random.rand()
                    b = np.random.rand()
                    c = 2.0*np.random.rand()+1
                    d = 2.0*np.random.rand()+1
                    e = -0.03+0.06*np.random.rand()
                    f = -0.03+0.06*np.random.rand()
                    samp_tmp[0,start_tmp:end_tmp] = start_pt[0] + (end_pt[0]-start_pt[0])*np.sin(np.pi/2*np.linspace(0,1,range_tmp))**a + e*np.sin(c*np.pi*np.pi*np.linspace(0,1,range_tmp))*np.sin(np.pi*np.linspace(0,1,range_tmp))
                    samp_tmp[1,start_tmp:end_tmp] = start_pt[1] + (end_pt[1]-start_pt[1])*np.sin(np.pi/2*np.linspace(0,1,range_tmp))**b + f*np.sin(d*np.pi*np.pi*np.linspace(0,1,range_tmp))*np.sin(np.pi*np.linspace(0,1,range_tmp))
                    # random trajectory between 'start' and 'end'

            samples.append(samp_tmp)

        return mean_traj, samples
    
    def showEnv(self,mean_traj,samples,showstart=False):

        for ss in range(len(samples)):
            plt.plot(samples[ss][0,:],samples[ss][1,:],color='gray',linewidth=.1)
        
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot(mean_traj[0,:],mean_traj[1,:],color='blue',linewidth=3)

        if showstart:
            plt.scatter(mean_traj[0,0],mean_traj[1,0],color='black',s=20)

        plt.show()



def likelihoodplotter(episode_likelihoods,assist_names):
    fig, ax = plt.subplots(2,1)
    num_pts = np.shape(episode_likelihoods)[0]
    for ii in range(np.shape(episode_likelihoods)[1]):
        ax[0].plot(np.arange(num_pts),np.array(episode_likelihoods)[:,ii])
    ax[0].legend(assist_names)
    ax[1].set_xlabel("Sample")
    ax[0].set_ylabel("Likelihood")
    
    # temporal plot of most likely assistance
    best = np.argmax(episode_likelihoods,axis=1)
    colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    for tt in range(num_pts):
        ax[1].bar(tt,1,1,color=colors[best[tt]])   
    plt.show()

def testLikelihoods():
    beta = 0.001**2.0
    sigmoid = 0.001
    alpha = [1.0]*4
    assisters = [NoAssist(beta=beta,sigmoid=sigmoid,alpha=alpha[0]), Teleoperation(beta=beta,sigmoid=sigmoid,alpha=alpha[1]), Discrete(k=2,beta=beta,sigmoid=sigmoid,alpha=alpha[2]), Corrections(k=1,beta=beta,sigmoid=sigmoid,alpha=alpha[3])]
    assist_names = []
    for assister in assisters:
        assist_names.append(assister.__class__.__name__)

    uc = Uncerpentine()
    num_samples = 100
    mean_traj, samples = uc.generateEnv(length=500,uncertainty_profiles=['teleoperation','discrete','correction'],starts=[50, 200, 350],ends=[150, 300, 450],num_samples=num_samples)
    samples = np.array(samples)

    rand_samp = np.random.choice(num_samples)
    uc.showEnv(samples[rand_samp],samples)

    samples_mod = np.transpose(samples,axes=[0,2,1])

    episode_liklihoods = []
    window = 1
    for tt in tqdm(range(500-window)):
        likelihoods = []

        if tt==250:
            print("hi")

        for assist in assisters:
            likelihoods.append(assist.getSumEntropy(samples_mod[:,tt:tt+window,:]))
  
        
        episode_liklihoods.append(likelihoods)
    likelihoodplotter(episode_liklihoods,assist_names=assist_names)

def unitTest():
    uc = Uncerpentine()
    num_samples = 50
    mean_traj, samples = uc.generateEnv(length=500,uncertainty_profiles=['teleoperation','discrete','correction'],starts=[50, 200, 350],ends=[150, 300, 450],num_samples=num_samples)
    rand_samp = np.random.choice(num_samples)
    print(np.shape(samples))
    uc.showEnv(samples[rand_samp],samples)

########## Next 3 are for setting things up ###############

def setUpEnv(save_location,env_id,num_samples=500):
     # assume the penalization and hmm weights are given
    num_tests = 5
    length = 500
    start_ranges = [50,200,350]
    end_start_ranges = [100,250,400]
    length_min = 50
    length_max = 99
  

    ######### CREATE ENVIRONMENT ############
    # sample order of methods and times when they start/stop
    uncertain_profiles = ['teleoperation','discrete','correction']
    np.random.shuffle(uncertain_profiles)

    starts = []
    ends = []
    for ii in range(3): # 3 uncertain events in behavior
        starts.append(np.random.randint(start_ranges[ii],end_start_ranges[ii]))
        ends.append(starts[ii]+np.random.randint(length_min,length_max))
    
    # create environment
    uc = Uncerpentine()
    total_samples = num_samples+num_tests
    mean_traj, samples = uc.generateEnv(length=length,uncertainty_profiles=uncertain_profiles,starts=starts,ends=ends,num_samples=total_samples)
    # draw n (e.g., 5) non-repeating samples from the training data
    test_samples = np.random.choice(total_samples, num_tests, replace=False)
    train_samples = list(set(range(total_samples)) - set(test_samples))

    # save environment
    env_file = save_location + str(env_id)+"_envdata.pkl"
    data_tmp = (uc,mean_traj,samples,train_samples,test_samples,starts,ends,uncertain_profiles)
    with open(env_file, 'wb') as handle:
        pickle.dump(data_tmp, handle)

    return data_tmp

def trainEnv(data_tmp,save_location,env_id):
    (uc,mean_traj,samples,train_samples,test_samples,starts,ends,uncertain_profiles) = data_tmp
    ######### TRAIN POLICY ############
    states, actions = uc.getStatesActions(samples,train_samples)
    policy = Diffusion(num_epochs=1, pred_horizon=64, diffusion_iterations=10)
    policy.train(states, actions)
    pol_file = save_location + str(env_id) + "_diffusion.pkl"
    policy.save(pol_file)
    return policy

def likEnv(data_tmp,policy,alpha,save_location,env_id):
    beta = 0.001**2.0
    sigmoid = 0.001


    (uc,mean_traj,samples,train_samples,test_samples,starts,ends,uncertain_profiles) = data_tmp

    na = NoAssist(beta=beta, sigmoid=sigmoid, alpha=alpha[0], num_actions=policy.action_dim, horizon=policy.estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)
    tele = Teleoperation(beta=beta, sigmoid=sigmoid, alpha=alpha[1], num_actions=policy.action_dim, horizon=policy.estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)
    disc = Discrete(k=2, beta=beta, sigmoid=sigmoid,alpha=alpha[2], num_actions=policy.action_dim, horizon=policy.estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)
    corr = Corrections(k=1, beta=beta, sigmoid=sigmoid,alpha=alpha[3], num_actions=policy.action_dim, horizon=policy.estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)

    assisters = [na, tele, disc, corr]

    # march along these samples and calculate likelihoods
    test_likelihoods = []
    test_penalized_likelihoods = []
    test_forecasts = []

    for ii, test_sample in enumerate(test_samples):
        # print(np.shape(samples[test_sample]))
        print("Testing ",ii," of ",len(test_samples))
        likelihoods = []
        penalized_likelihoods = []
        forecasts = []
        for tt in tqdm(range(np.shape(samples[test_sample])[1])): # 2xlength

            forecast = np.array(policy.forecastAction(samples[test_sample][:,tt],50,policy.estimate_horizon,None)) # last arg is env which is not used for diffusion
            forecasts.append(forecast)

            likelihoods_tmp = []
            penalized_likelihoods_tmp = []
            for assister in assisters:
                likelihoods_tmp.append(assister.getLikelihood(forecast))
                penalized_likelihoods_tmp.append(assister.getPenalizedLikelihood(forecast))
            likelihoods.append(likelihoods_tmp)
            penalized_likelihoods.append(penalized_likelihoods_tmp)
        test_likelihoods.append(likelihoods)
        test_penalized_likelihoods.append(penalized_likelihoods)
        test_forecasts.append(forecasts)
        # uc.showEnv(samples[test_sample],samples)

    # save everything to pkl: environment, samples, drawn samples, likelihoods
    lik_file = save_location + str(env_id)+"_likelihoods.pkl"
    data_tmp2 = (test_forecasts,test_likelihoods,test_penalized_likelihoods)
    with open(lik_file, 'wb') as handle:
        pickle.dump(data_tmp2, handle)

    return data_tmp2


def getPlot(test_traj,samples,likelihoods,penalized_likelihoods,assist_names,save_location,env_id,indtmp):
    fig, ax = plt.subplots(3,1,figsize=(4,5),gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot uncerpentine
    for ss in range(len(samples)):
        ax[0].plot(samples[ss][0,:],samples[ss][1,:],color='gray',linewidth=.1)
    
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].plot(test_traj[0,:],test_traj[1,:],color='blue',linewidth=3)

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Plot likelihoods and penalized likelihoods
    
    num_pts = np.shape(likelihoods)[0]
    for ii in range(np.shape(likelihoods)[1]):
        ax[1].plot(np.arange(num_pts),np.array(likelihoods)[:,ii])
    # ax[1].legend(assist_names)
    ax[1].set_ylabel("Lik.")

    num_pts = np.shape(penalized_likelihoods)[0]
    for ii in range(np.shape(penalized_likelihoods)[1]):
        ax[2].plot(np.arange(num_pts),np.array(penalized_likelihoods)[:,ii])
    # ax[2].legend(assist_names)
    ax[2].set_ylabel("Pen. Lik.")

    ax[2].set_xlabel("Sample")

    # plt.show()
    plt.savefig(save_location+str(env_id)+"_plot_"+str(indtmp)+'.png')

def getPlots(data_tmp,data_tmp2,save_location,env_id):
    (uc,mean_traj,samples,train_samples,test_samples,starts,ends,uncertain_profiles) = data_tmp
    (test_forecasts,test_likelihoods,test_penalized_likelihoods) = data_tmp2

    num_tests = len(test_samples)

    samples_train = []
    for ll in range(len(train_samples)):
        samples_train.append(samples[train_samples[ll]])

    indtmp = 0
    assist_names = ["No Assist","Teleoperation","Discrete","Corrections"]
    getPlot(samples[test_samples[indtmp]],samples_train,test_likelihoods[indtmp],test_penalized_likelihoods[indtmp],assist_names,save_location,env_id,indtmp)


def calculateMetrics(policy,data_tmp, data_tmp2, save_location, env_id, print_metrics=True):
    (uc,mean_traj,samples,train_samples,test_samples,starts,ends,uncertain_profiles) = data_tmp
    (test_forecasts,test_likelihoods,test_penalized_likelihoods) = data_tmp2

    num_tests = len(test_samples)

    gts = []
    actuals = []
    for ll in range(num_tests):
        num_samples = len(test_likelihoods[ll])
        gt = []
        actual = []

        for tt in range(num_samples):
            
            gt_tmp = 0
            for ii in range(3):
                # no assist, teleperation, discrete, corrections (order)
                if tt > starts[ii] and tt < ends[ii]:
                    type_tmp = uncertain_profiles[ii]
                    if type_tmp=='teleoperation':
                        map_tmp = 2
                    if type_tmp=='discrete':
                        map_tmp = 3
                    if type_tmp=='correction':
                        map_tmp = 1
                    gt_tmp = map_tmp

            actual_tmp = np.argmax(test_penalized_likelihoods[ll][tt])

            # mapping flip to no assist, correction, teleoperation for figure table
            actual_map = [0,2,3,1]
            actual_tmp = actual_map[actual_tmp]

            gt.append(gt_tmp)
            actual.append(actual_tmp)
        gts.append(gt)
        actuals.append(actual)

    # 3 x 4 confusion (stored as 4x4 -- discrete)
    confusion = np.zeros((4,4))
    for ll in range(num_tests):
        for tt in range(num_samples):
            confusion[gts[ll][tt],actuals[ll][tt]]+=1
    
    if print_metrics:
        print("CONFUSION:")
        row_sums = confusion.sum(axis=1)
        print(confusion)
        print(row_sums)

        print(confusion / row_sums[:, np.newaxis])

    success_discrete = 0
    total_discrete = 0
    for ll in range(num_tests):
        disc_id = uncertain_profiles.index("discrete")

        disc_found = False
        for tt in range(starts[disc_id]-policy.estimate_horizon,starts[disc_id]):
            if np.argmax(test_penalized_likelihoods[ll][tt])==2: # index 2 in the original mapping
                disc_found = True
        
        if disc_found:
            success_discrete+=1
        total_discrete+=1

    if print_metrics:
        print("Discrete Found: ",success_discrete," of ",total_discrete)

    if save_location is not None:
        metrics_file = save_location + str(env_id)+"_metrics.pkl"
        data_tmp3 = (confusion, success_discrete, total_discrete, gts, actuals)
        with open(metrics_file, 'wb') as handle:
            pickle.dump(data_tmp3, handle)

    return confusion, success_discrete, total_discrete, gts, actuals


def testEnvironment(alpha,gamma,save_location,env_id,num_samples=500):

    # STEP 0: create environment
    data_tmp = setUpEnv(save_location,env_id,num_samples)
    with open(save_location+str(env_id)+"_envdata.pkl", 'rb') as handle:
        data_tmp = pickle.load(handle)


    # STEP 1: learn policy
    policy = trainEnv(data_tmp,save_location,env_id) 
    # these next two lines are required (even if training)
    policy = Diffusion()
    policy.load(save_location+str(env_id)+"_diffusion.pkl")

    # Step 2: assess policy
    data_tmp2 = likEnv(data_tmp,policy,alpha,save_location,env_id)
    with open(save_location+str(env_id)+"_likelihoods.pkl", 'rb') as handle:
        data_tmp2 = pickle.load(handle)


    # Step 3: calculate metrics
    metrics = calculateMetrics(policy,data_tmp, data_tmp2, save_location, env_id)

    # Step 4: generate plots
    getPlots(data_tmp,data_tmp2,save_location,env_id)





if __name__ == "__main__":
   
    # no assist, teleoperation, discrete, corrections
    delta = 0.0001
    alpha = [1.0, 1-3*delta, 1-1*delta, 1-2*delta]
    gamma = 1.0
    save_location = project_path+'/saved_policies/simulated/'
    
    testEnvironment(alpha,gamma,save_location,20,num_samples=1000)
