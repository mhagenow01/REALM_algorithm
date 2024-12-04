## <b>REALM</b> -- Real-Time Estimation of assistance for Learned Models in Human-Robot Interaction (Supplemental Code)

This repository contains basic code related to the assistance mechanisms and the environment used in the simulated/manipulation user study tasks. Upon acceptance, we plan to release a larger, cleaned up github repository (where it would be difficult to maintain anonyminity). Our hope is that this limited release will help researchers understand our environment in greater detail and the specific calculations/packages that are used to calculate the post-intervention entropy estimates.

### Contents
* assistance
    * base_assistance.py - abstract base class for the different assistance mechanisms.
    * corrections.py - class that demonstrates the SVD/PCA approach to latent-space input.
    * discrete.py - class that demonstrates the clustering approach to identify discrete behaviors.
    * no_assist.py - class that computes the sample-based entropy for when no assistance is given to the robot.
    * teleoperation.py - class for teleoperation which is an analytic entropy expression for the expected human error.
* envs
    * uncerpentine.py - 2D navigation environment used in the simulated study. Contains definition for the randomized environment as well as the learning/assessment tools (though the Diffusion policy code is not included in this limited release).
    * ur5e_tending.py - manipulation environment for our user study task. Includes generation of simulated data as well as the ROS pipeline that ultimately issues commands to the robot (the robot code is not included in this limited release).