import numpy as np
import pickle
import os
import sys

import time
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
project_path = dir_path+'/..'

sys.path.append(project_path+'/policies')
sys.path.append(project_path)

# handle exception for ros messages when training without ros
try:
    import rospy
    import tf2_ros

    from std_msgs.msg import Float64, Bool, String
    from geometry_msgs.msg import WrenchStamped, PoseStamped, TransformStamped

    from visualization_msgs.msg import Marker, MarkerArray
except:
    pass

try:
    from policies.diffusion import Diffusion
except:
    pass

from scipy.spatial.transform import Rotation as ScipyR
from scipy.spatial.transform import Slerp

pkg_path = os.path.dirname(os.path.abspath(__file__))


def circ_marker(id, pos, size, color=[0.0, 1.0, 0.0], frame="map"):
    """ Creates circular markers on points of interest """
    pos_tmp = pos.copy()
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "pts"
    marker.id = id
    marker.type = 2 # sphere
    marker.action = 0 # Add/Modify
    marker.pose.position.x = pos_tmp[0]
    marker.pose.position.y = pos_tmp[1]
    marker.pose.position.z = pos_tmp[2]
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
    marker.color.a = 1
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    return marker

class PriviledgedPolicy():
    def __init__(self,workpiece_loc=np.array([-0.092,-0.339,0.019]),jig_rotation=1*np.pi/6.0,blue_bin=True,optimal=False) -> None:

        self.index = 0
        # self.starting_pos = obs['robot0_eef_pos']
        self.starting_pos = np.array([-0.2, -0.30, 0.15]) # force to start in same location
        self.starting_q = np.array([0,0,-0.7071,0.7071])

        r_addtl = ScipyR.from_rotvec([0,0,0])
        R_starting = ScipyR.from_quat(self.starting_q)
        self.starting_q = (r_addtl*R_starting).as_quat()

        self.starting_rotvec = ScipyR.from_quat(self.starting_q).as_rotvec()
        self.prev_action = None
        
        # used in generating behavior
        self.workpiece_loc = workpiece_loc
        self.blue_bin = blue_bin
        self.jig_rotation = jig_rotation

        self.above_starting_pt = np.array([-0.08,-0.34,0.15])

        self.jig_pos = np.array([-0.0032, -0.442, 0.0279])
        self.above_jig = np.array([-0.0032, -0.442, 0.15])
        self.right_of_jig = np.array([0.108,-0.442, 0.122])
        self.right_forward = np.array([0.108,-0.410, 0.122])

        self.blue_bin_loc = np.array([0.153, -0.274, 0.108])
        self.red_bin_loc = np.array([0.052, -0.274, 0.108])

        self.optimal = optimal

        self.stage_times = [20,60,80,130,170,200,220,230,260,330]
        self.episode_length = self.stage_times[-1]

    def getAction(self,obs):

        # print(self.index)
        if self.prev_action is None:
            action = np.array([self.starting_pos[0], self.starting_pos[1], self.starting_pos[2], self.starting_q[0], self.starting_q[1], self.starting_q[2], -1])
        else:
            action = self.prev_action

        stage_times = self.stage_times

        #########################################################################
        # Below are the simulated rollout behaviors with sinusoidal modulations #
        #########################################################################

        if self.index < stage_times[0]: # mode 0: above starting point
            action[0:3] = self.starting_pos + float(self.index)/stage_times[0]*(self.above_starting_pt-self.starting_pos)
            action[3:6] = self.starting_rotvec
            action[6] = -1 # gripper open
        elif self.index < stage_times[1]: # mode 1: prepare to grasp
        
            if self.index==stage_times[0]:
                self.rand1 = 1.5*np.random.rand()+0.5 # 0.5 to 2.0
                self.rand2 = 1.5*np.random.rand()+0.5 # 0.5 to 2.0
                self.rand3 = 1.5*np.random.rand()
                self.rand4 = 1.5*np.random.rand()
                self.rand5 = 0.02*np.random.rand()
                self.rand6 = 0.02*np.random.rand()

                # approach from front, top, or back
                self.approach = np.random.randint(-1,2)
                self.approach_amp = 0.05*float(self.approach)
              
            tmp_index = self.index - stage_times[0]
            stage_len = stage_times[1]-stage_times[0]

            above_starting_pt = self.above_starting_pt
            if self.optimal:
                action[0:3] = above_starting_pt + float(tmp_index/stage_len)*(self.workpiece_loc-above_starting_pt)
            else:
                action[0] = above_starting_pt[0] + float(tmp_index < stage_len*0.8)*self.rand5*np.sin(np.pi*(float(tmp_index)/(stage_len*0.8)))*np.sin(self.rand3*np.pi*(float(tmp_index)/stage_len)) + np.sin(np.pi/2*(float(tmp_index)/stage_len)**self.rand1)*(self.workpiece_loc[0]-above_starting_pt[0])
                action[1] = above_starting_pt[1] + float(tmp_index < stage_len*0.8)*self.rand6*np.sin(np.pi*(float(tmp_index)/(stage_len*0.8)))*np.sin(self.rand4*np.pi*(float(tmp_index)/stage_len)) + np.sin(np.pi/2*(float(tmp_index)/stage_len)**self.rand2)*(self.workpiece_loc[1]-above_starting_pt[1]) + self.approach_amp*np.sin(np.pi*(float(tmp_index)/stage_len)**1.0)
                action[2] = above_starting_pt[2] + float(tmp_index)/stage_len*(self.workpiece_loc[2]-above_starting_pt[2])
            action[3:6] = self.starting_rotvec
            action[6] = -1 # gripper open

        elif self.index < stage_times[2]: # mode 2: grasp
            tmp_index = self.index - stage_times[1]
            stage_len = stage_times[2]-stage_times[1]

            if self.index==stage_times[1]:
                self.start_grip_sample = np.random.randint(0,int(stage_len*0.8))
            tmp_index = self.index - stage_times[1]
            action[0:3] = self.workpiece_loc
            action[3:6] = self.starting_rotvec
            if self.optimal:
                if tmp_index>(stage_len/2):
                    action[6] = 1
                else:
                    action[6] = -1
            else:
                if tmp_index > self.start_grip_sample:
                    action[6] = 1
                else:
                    action[6] = -1

        elif self.index < stage_times[3]: # mode 3: move above jig
            tmp_index = self.index - stage_times[2]
            stage_len = stage_times[3]-stage_times[2]
            
            # optimal and normal are the same -- less variability after grasping
            action[0:3] = self.workpiece_loc + float(tmp_index)/stage_len*(self.above_jig-self.workpiece_loc)
            ind_tmp = min(1,float(tmp_index)/(stage_len/1.5))
            action[2] = self.workpiece_loc[2] + ind_tmp*(self.above_jig[2]-self.workpiece_loc[2])

            action[3:6] = self.starting_rotvec
            action[6] = 1 # gripper closed

        elif self.index < stage_times[4]: # mode 4: move into jig
                 
            tmp_index = self.index - stage_times[3]
            stage_len = stage_times[4]-stage_times[3]

            if self.index==stage_times[3]:
                self.rand1 = 1.5*np.random.rand()+0.5 # 0.5 to 2.0
                self.rand2 = 1.5*np.random.rand()+1.5 # 1.5 to 3.0
                self.rand3 = 0.2*np.random.rand()+0.1

            scipy_starting_q = ScipyR.from_rotvec(self.starting_rotvec)
            jig_orientation = ScipyR.from_rotvec([0.0, 0.0, self.jig_rotation])

            self.desired_q = jig_orientation * scipy_starting_q
            slerper = Slerp([0,1],ScipyR.concatenate((scipy_starting_q,self.desired_q )))
            action[0:3] = self.above_jig + float(tmp_index)/stage_len*(self.jig_pos-self.above_jig)
           

            if self.optimal:
                traj_tmp = float(tmp_index)/(0.25*stage_len)
                traj_tmp = min(1,traj_tmp)
            else:
                traj_tmp = np.sin(np.pi/2*(float(tmp_index)/stage_len)**self.rand1)+float(tmp_index < (stage_len*0.8))*self.rand3*np.sin(np.pi*(float(tmp_index)/(stage_len*0.8)))*np.sin(self.rand2*np.pi*(float(tmp_index)/(stage_len*0.8)))
                traj_tmp = min(1,traj_tmp) # saturate in case over bounds
                traj_tmp = max(0,traj_tmp)
            action[3:6] = slerper(traj_tmp).as_rotvec() # negative because gripper points down and workpiece points up (z axis)
            if self.index == stage_times[4]-1:
                self.jigged_pos = action[0:3]
        
        elif self.index < stage_times[5]: # mode 5: move above jig
            tmp_index = self.index - stage_times[4]
            stage_len = stage_times[5]-stage_times[4]

            scipy_starting_q = ScipyR.from_rotvec(self.starting_rotvec)
            jig_orientation = ScipyR.from_rotvec([0.0, 0.0, self.jig_rotation])
            
            self.desired_q = jig_orientation * scipy_starting_q
            slerper = Slerp([0,1],ScipyR.concatenate((self.desired_q,scipy_starting_q)))
            
            action[0:3] = self.jigged_pos + float(tmp_index)/stage_len*(self.above_jig-self.jigged_pos)
            
            # optimal and random are the same 
            slerp_tmp = max(0,-0.2+1.2*float(tmp_index)/stage_len)
            action[3:6] = slerper(slerp_tmp).as_rotvec()
           

        elif self.index < stage_times[6]: # mode 6: move right
            tmp_index = self.index - stage_times[5]
            stage_len = stage_times[6]-stage_times[5]

            action[0:3] = self.above_jig + float(tmp_index)/stage_len*(self.right_of_jig-self.above_jig)
            action[3:6] = self.starting_rotvec

        elif self.index < stage_times[7]: # mode 7: move forward toward bin split
            tmp_index = self.index - stage_times[6]
            stage_len = stage_times[7]-stage_times[6]

            action[0:3] = self.right_of_jig + float(tmp_index)/stage_len*(self.right_forward-self.right_of_jig)
            action[3:6] = self.starting_rotvec

        elif self.index < stage_times[8]: # mode 8: move to basket
            tmp_index = self.index - stage_times[7]
            stage_len = stage_times[8]-stage_times[7]

            if self.blue_bin:
                end_pt = self.blue_bin_loc
            else:
                end_pt = self.red_bin_loc
            
            # optimal and normal are the same
            action[0:3] = self.right_forward + float(tmp_index)/stage_len*(end_pt-self.right_forward)
            action[0] = self.right_forward[0] + np.sin(np.pi/2*(float(tmp_index)/stage_len))**0.6*(end_pt[0]-self.right_forward[0])
            action[3:6] = self.starting_rotvec

            action[6] = 1 # gripper closed
        
        elif self.index < stage_times[9]: # mode 9: drop block
            action = self.prev_action
            action[6] = -1 # gripper open

        self.index +=1
        self.prev_action = action.copy()
        return action

    def actionToObs(self,action):
        tended = int(self.index >= self.stage_times[4])
        return action[0], action[1], action[2], action[3], action[4], action[5], action[6], tended 


class UR5ETending():
    def __init__(self,stateaction_save_file = None) -> None:       
        self.jig_set = False
        self.index = 0
        self.half_green_count = 0
        self.gripper = -1
        self.stateaction_save_file = stateaction_save_file
        self.prev_command = None

    def storePose(self, data):
        self.curr_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.curr_q = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])
 
    def getOptimalAction(self):
        return self.pp.getAction(None)

    def step(self,action,meta=None,full=False):
        self.index+=1

        if meta is not None:
            try:
                names = meta['assist_names']
                likelihoods = np.array(meta['assist_likelihoods'])
                likelihoods_raw = np.array(meta['likelihoods_raw'])

                teleop_active = False
                discrete_active = False
                if 'teleoperation_active' in meta.keys():
                    teleop_active = meta['teleoperation_active']
                if 'discrete_active' in meta.keys():
                    discrete_active = meta['discrete_active']

                if 'sm_state' in meta.keys():
                    sm_string = json.dumps(meta['sm_state'])
                    self.humaninput_pub.publish(String(sm_string))

                likelihoods_str = " ".join(str(x) for x in likelihoods)
                self.likelihood_pub.publish(String(likelihoods_str))
                likelihoods_raw_str = " ".join(str(x) for x in likelihoods_raw)
                self.likelihoodraw_pub.publish(String(likelihoods_raw_str))

                color_tmp = [120,120,120]
                assister = names[np.argmax(likelihoods)]
                if discrete_active:
                    self.led_pub.publish(String("p"))
                    color_tmp = [255,0,0]
                else:                
                    if assister=='Discrete':
                        color_tmp = [255,0,0]
                        if teleop_active:
                            self.led_pub.publish(String("f"))
                        else:
                            self.led_pub.publish(String("r"))
                    elif assister=='Corrections':
                        color_tmp = [255,255,0]
                        if teleop_active:
                            self.led_pub.publish(String("h"))
                        else:
                            self.led_pub.publish(String("y"))
                    elif assister=='Teleoperation':
                        if self.half_green_count < 3:
                            color_tmp = [120,120,120]
                            self.led_pub.publish(String("d"))
                            self.half_green_count+=1
                        else:
                            color_tmp = [0,255,0]
                            self.led_pub.publish(String("g"))
                    else: # no assist
                        color_tmp = [120,120,120]
                        if teleop_active:
                            self.half_green_count = 0
                            self.led_pub.publish(String("d"))
                        else:
                            self.led_pub.publish(String("w"))

                if 'forecasted_actions' in meta.keys():
                    self.showForecastRViz(meta['forecasted_actions'],color_tmp)

            except Exception as e:
                print(e)

        ####### Take the action

        R_tmp = ScipyR.from_rotvec(action[3:6])
        q_tmp = R_tmp.as_quat()

        if self.prev_command is not None:
            rot_vec_dist = np.linalg.norm(action[3:6]-self.prev_command[3:6])
            lin_dist = np.linalg.norm(action[0:3]-self.prev_command[0:3])
            if lin_dist>0.05 or rot_vec_dist>0.3:
                raise Exception("MOVED TO FAR IN ONE MOVE- Lin:"+str(lin_dist)+" Rot:"+str(rot_vec_dist))

        

        pose_out = PoseStamped()
        pose_out.header.frame_id = 'map'
        pose_out.header.stamp = rospy.Time.now()
        pose_out.pose.position.x = action[0]
        pose_out.pose.position.y = action[1]
        pose_out.pose.position.z = action[2]
        pose_out.pose.orientation.x = q_tmp[0]
        pose_out.pose.orientation.y = q_tmp[1]
        pose_out.pose.orientation.z = q_tmp[2]
        pose_out.pose.orientation.w = q_tmp[3]
        self.pose_pub.publish(pose_out)

        if action[6]<0.0:
            self.gripper_pub.publish(Bool(False))
        else:
            self.gripper_pub.publish(Bool(True))

        self.gripper = action[6] # use for current gripper observation

        if np.linalg.norm(self.pp.jig_pos[0:2] - self.curr_pos[0:2]) < 0.02 and np.linalg.norm(self.pp.jig_pos[2] - self.curr_pos[2])<0.012:
            self.jig_set = True 

        self.prev_command = action.copy()
        return self.getTruncatedObservation(), 0.0, self._check_terminated(), None

        return (obs,step_rtn[1],self._check_terminated(),step_rtn[3])


    def showForecastRViz(self,forecast,color_tmp):
        # forecast is num_samples x horizon x state_vars
        num_samples = np.shape(forecast)[0]
        horizon = np.shape(forecast)[1]
        downsamp = 1

        markerarr = MarkerArray()
        plotting_size = 0.003

        for ii in range(0,num_samples):
            for jj in range(0,horizon,downsamp):
                pos = forecast[ii][jj,0:3]
                markerarr.markers.append(circ_marker(ii*horizon+jj, pos.copy(), plotting_size, color=[float(color_tmp[0])/255, float(color_tmp[1])/255, float(color_tmp[2])/255], frame="base"))
        
        # publish marker array representing path
        self.forecastMarkerpub.publish(markerarr)


    def reset(self,full=False):
        # set up ROS topics
        try:
            rospy.init_node('ur5e_tending')
        except Exception as e:
            pass

        self.pp =  PriviledgedPolicy(optimal=True)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        self.pose_pub = rospy.Publisher('/ur5e/compliant_pose', PoseStamped, queue_size=1)
        self.gripper_pub = rospy.Publisher('/ur5e/gripper_command', Bool, queue_size=1)

        self.led_pub = rospy.Publisher('/led_state', String, queue_size=1)

        self.ik_pub = rospy.Publisher('/ur5e/desired_pose', PoseStamped, queue_size=1)

        self.humaninput_pub = rospy.Publisher('/humaninput', String, queue_size=1)
        self.likelihood_pub = rospy.Publisher('/likelihoods', String, queue_size=1)
        self.likelihoodraw_pub = rospy.Publisher('/likelihoodsraw', String, queue_size=1)

        self.biasFTpub = rospy.Publisher('/ur5e/biasFT', String, queue_size=1)
        self.forecastMarkerpub = rospy.Publisher("/forecastmarkerpub", MarkerArray, queue_size =1, latch = True)

        rospy.Subscriber("/ur5e/desired_pose", PoseStamped, self.storePose)
        time.sleep(1)
        
        rate = rospy.Rate(1)
        curr_pos_acquired = False
        self.jig_set = False

        while not curr_pos_acquired:
            try:
                trans = self.tfBuffer.lookup_transform('base', 'toolnew', rospy.Time())
                curr_pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
                curr_q = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
                curr_pos_acquired = True
            except Exception as e:
                print(e)
            rate.sleep()

        # Step 0: ungrip
        self.led_pub.publish(String("s"))
        self.gripper_pub.publish(Bool(False))
        time.sleep(1.5)

        # Step 1: go up
        des_z = 0.15
        dist_tmp = np.linalg.norm(des_z-curr_pos[2])
        num_steps = max(1,int(10*(dist_tmp/0.05))) # 5cm/s


        rate = rospy.Rate(10)

        for tt in range(num_steps):
            interp_pos = curr_pos + float(tt)/num_steps*(self.pp.starting_pos-curr_pos)
            pose_out = PoseStamped()
            pose_out.header.frame_id = 'map'
            pose_out.header.stamp = rospy.Time.now()
            pose_out.pose.position.x = curr_pos[0]
            pose_out.pose.position.y = curr_pos[1]
            pose_out.pose.position.z = curr_pos[2]+float(tt)/num_steps*(des_z-curr_pos[2])
            pose_out.pose.orientation.x = curr_q[0]
            pose_out.pose.orientation.y = curr_q[1]
            pose_out.pose.orientation.z = curr_q[2]
            pose_out.pose.orientation.w = curr_q[3]
            self.pose_pub.publish(pose_out)
            rate.sleep()

        curr_pos[2] = des_z

        # Step 2: interpolate and rotate
        dist_tmp = np.linalg.norm(self.pp.starting_pos-curr_pos)
        num_steps = max(1,int(10*(dist_tmp/0.1))) # 10cm/s
        
        key_rots = ScipyR.from_quat([curr_q,self.pp.starting_q])
        slerper = Slerp([0,num_steps],key_rots)


        for tt in range(num_steps):
            interp_pos = curr_pos + float(tt)/num_steps*(self.pp.starting_pos-curr_pos)
            interp_q = slerper(tt).as_quat()
            pose_out = PoseStamped()
            pose_out.header.frame_id = 'map'
            pose_out.header.stamp = rospy.Time.now()
            pose_out.pose.position.x = interp_pos[0]
            pose_out.pose.position.y = interp_pos[1]
            pose_out.pose.position.z = interp_pos[2]
            pose_out.pose.orientation.x = interp_q[0]
            pose_out.pose.orientation.y = interp_q[1]
            pose_out.pose.orientation.z = interp_q[2]
            pose_out.pose.orientation.w = interp_q[3]
            self.pose_pub.publish(pose_out)
            rate.sleep()

        self.index = 0
        self.pp.index = 0
        self.prev_command = None

        self.led_pub.publish(String("e"))
        time.sleep(1.0)

        self.biasFTpub.publish(String("bias"))
        time.sleep(0.5)

        
        return self.getTruncatedObservation()

        
    def render(self):
        pass # no rendering in real life

    def reward(self, action=None):
        """
        Reward function for the task.

        float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1

        # use a shaping reward
        elif self.reward_shaping:
            reward = 0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward


    def _check_success(self):
        """
        Returns:
            bool: True if piece has been put in the jig and deposited in the correct bin
        """

        return False

    def _check_terminated(self):
        if self.jig_set:
            if self.gripper==-1:
                if np.linalg.norm(self.curr_pos - self.pp.blue_bin_loc)<0.01 or np.linalg.norm(self.curr_pos - self.pp.red_bin_loc)<0.01:
                    self.led_pub.publish(String("e"))
                    return True
        return False
    
    def pygameKeyboard(self):
        return False

    def makeData(self,num_trajs):
        states = []
        actions = []
        print("\nUR5e_tending: generating data for ",num_trajs," trajectories!")
        ii = 0
        while ii < num_trajs:
            
            # decide workpiece_loc, red or blue, and angle of rotation
            blue_bin = np.random.rand()>0.5
            jig_rotation = 2.0*(np.random.rand()-0.5)*np.pi

            workpiece_loc=np.array([-0.092,-0.339,0.019])
            workpiece_loc[0]+=(-0.03+np.random.rand()*(0.01-(-0.03))) # -3cm to +1 cm
            workpiece_loc[1]+=(-0.03+np.random.rand()*(0.00-(-0.03))) # -3cm to +0 cm

            print("..trajectory ",ii,np.round(workpiece_loc,2),np.round(jig_rotation,2),blue_bin)

            pp = PriviledgedPolicy(workpiece_loc=workpiece_loc,jig_rotation=jig_rotation, blue_bin=blue_bin)
            tt=0
            episode_len = pp.episode_length-1
            states_tmp = np.zeros((8,episode_len)) # pos, rotvec, gripper, tended
            actions_tmp = np.zeros((7,episode_len)) # pos, rotvec, gripper
            for tt in range(1,episode_len):
                action_temp = pp.getAction(None)
                states_tmp[:,tt] = pp.actionToObs(action_temp) + np.random.normal(0,0.001,8)
                actions_tmp[:,tt-1] = action_temp + np.random.normal(0,0.001**2.0,7) # noise modification
    
            states.append(states_tmp.copy())
            actions.append(actions_tmp.copy())
            ii+=1
        print("UR5e_tending: finished generating data")
        return states, actions
    
    def saveData(self,num_trajs=100):
        states, actions = self.makeData(num_trajs)
        if self.stateaction_save_file is not None:
            with open(self.stateaction_save_file, 'wb') as handle:
                pickle.dump((states, actions), handle)

    def getData(self,num_trajs=100):
        # states and actions
        # states: num_trajs x num_vars x time
        # actions: num_trajs x num_action_vars x time
        if self.stateaction_save_file is not None:
            with open(self.stateaction_save_file, 'rb') as handle:
                (states, actions) = pickle.load(handle)
        else:
            states, actions = self.makeData(num_trajs=num_trajs)
        
        return states, actions
    
    def forecastStep(self,state,action):
        # since the action is absolute, the returned state should just be the action (but converted to state format)
        new_state = state.copy()
        new_state[0:-1] = action
        new_state[-1] += 1 # increment index
        return new_state
    
    def getTruncatedObservation(self):
        # used in case there is a different (partially observable) observation used in the policy
        R_curr = ScipyR.from_quat(self.curr_q)
        rotvec_curr = R_curr.as_rotvec()

        return self.curr_pos[0], self.curr_pos[1], self.curr_pos[2],rotvec_curr[0], rotvec_curr[1], rotvec_curr[2], self.gripper, int(self.jig_set)
