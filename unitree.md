# Simulation and Teleop Setup
This documentation assumes a ROS2 distro (e.g. ROS2 Jazzy) is already installed on the computer.

## ROS2 in conda python 3.12
In case running and building ROS2 packages in a conda virtual environment fail:
- ROS2 only supports python 3.12, if a conda environment is used, it must be created with python 3.12:
  ```bash
  conda create -n myenv python=3.12
  ```
- Required parameter:
  ```bash
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  ```
  This line is also added to the unitree_setup.bash file.
  
- Required installation:
  ```bash
    pip install empy
    pip install catkin_pkg
    pip install lark lark-parser
  ```
  ROS2 packages previously built with a higher version python (if any) need to be rebuilt with python 3.12
  
## [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) 
- This repo is needed for the unitree ros2 interfaces

- Follow the setup instruction in the `README.md` file of the [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) repo, replace the ros2 distro name in every command to the correspondingly installed ros2 distro (for our case from foxy to jazzy), e.g.: sudo apt install ros-~~foxy~~jazzy-rmw-cyclonedds-cpp

- The following need to be done after the colcon build of cyclonedds_ws:
  1) For each of the files `unitree_ros2/setup.sh`, `unitree_ros2/setup_local.sh`, `unitree_ros2/setup_default.sh`:
     - Change all ros2 distro name in the following setup file to the correct distro (here from foxy to jazzy)
     - Change the `NetworkInterface name` in `CYCLONEDDS_URI` to the correct one:
       ```bash
        export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="wlp0s20f3" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
       ```
       
       To see the available network interfaces, enter in the command line:
       ```bash
       ip link show
       ```
  2) Source the `unitree_ros2/setup_local.sh` file to setup the simulation environment. 

- Only relevant for my team laptop, but can possibly be helpful:
  
  To fix the sudo apt update not working problem, /usr/share/keyrings/ros-archive-keyring.gpg was replaced by:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  ```
  The old key is stored under `/usr/share/keyrings/ros-archive-keyring_old.gpg`
    
- Errors and Fix:
    1) The command
       ```bash
       cd unitree_mujoco/simulate/build/
       ./unitree_mujoco simulation
       ```
       can cause crashing problem because of a DDS runtime mismatch.

       This can be fixed by
       ```bash
       export LD_LIBRARY_PATH=/opt/unitree_robotics/lib:$LD_LIBRARY_PATH
       ```
       before running the simulation. (This line is also added to the unitree_setup.bash file)
       
    2) After changing the Network interface name in
       ```bash
        export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="wlp0s20f3" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
       ```
        If `CYCLONEDDS_URI` seems not to be applied (e.g. topics from a previously connected network can still be seen, or `ros2 topic list` spins forever without output any topics when a previously connected network interface is no more in connection), reconnect to the previously connected network and restart ros2 daemon by:
       ```bash
        ros2 daemon stop
        ros2 daemon start
       ```
       This only works if the previously connected network interface is reconnected. If that network interface is no more available, reboot the computer and try again.
       
## ros_recording
Source the setup file:
```bash
ros_recording/install/setup.bash
```
for the sensor-suit's ROS2 interfaces

## unitree_setup.bash (Optional)
- For your convenience, the `unitree_setup.bash` file is prepared, which covers all setup steps in the [ROS2 in conda python 3.12](#ros2-in-conda-python-312), [unitree_ros2](#unitree_ros2) and [ros_recording](#ros_recording) sections. Source this after `colcon build` the two ros2 packages.
 
### Notes: 
- Both [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) and **ros_recording** need to be put under the same directory.
- This directory need to be set as the value of the `ROOT_DIR` parameter:
  ```bash
  # Define root directory (change this one if the root directory of unitree_ros2 is different)
    ROOT_DIR=~/Desktop/quan
  ```
-  If source fails after changing the Network interface name, a reboot is required (refer to the second fix under the [unitree_ros2](#unitree_ros2) section of this file)
       
       
## [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
### Download Repo
- Clone the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) repo and follow the installation instruction for the **C++ Simulator (simulate)** in the repo's README.md file
### Setup config.yaml file
- Copy the following text into the `unitree_mujoco/simulate/config.yaml` to setup the simulation's scene and robot:
  ```yaml
  robot: "g1"  # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
  robot_scene: "scene_23dof.xml" # Robot scene for the 23dof g1 (/unitree_robots/[robot]/scene.xml)
  
  domain_id: 1  # Domain id
  interface: "wlp0s20f3" # Interface, change to the same interface set in CYCLONEDDS_URI
  
  use_joystick: 0 # Simulate Unitree WirelessController using a gamepad
  joystick_type: "xbox" # support "xbox" and "switch" gamepad layout
  joystick_device: "/dev/input/js0" # Device path
  joystick_bits: 16 # Some game controllers may only have 8-bit accuracy
  
  print_scene_information: 1 # Print link, joint and sensors information of robot
  
  enable_elastic_band: 1 # Virtual spring band, used for lifting h1
  ```
- The `interface` parameter must be set to be the same network interface name in `CYCLONEDDS_URI` (compare [unitree_ros2](#unitree_ros2) section)
### Notes and Possible Errors
- The C++ simulation was tested and worked with the Mujoco version **3.3.6**
- If the compilation step (**2. Compile unitree_mujoco**) of the installation instruction for the **C++ Simulator (simulate)** in the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)'s `readme.md` does not succeed with the error:
  ```bash
  error: ‘uint8_t’ does not name a type
  ```
  Add `#include <cstdint>` to the beginning of the `unitree_mujoco/simulate/src/joystick/jstest.cc` file and try compiling again 
  
- Setup Python Simulation
  
  Not relevant for C++ Simulator, but I put it here in case Python Simulation is wanted in the future.
  - Possible error:
    ```bash
    Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
    ```
    _Solution_: Follow the solution provided in the installation instruction for the **Python Simulator (simulate_python)** in the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)'s `readme.md`, try:
    ```bash
    export CYCLONEDDS_HOME=~/cyclonedds/install
    ```
     if
    ```bash
    export CYCLONEDDS_HOME="~/cyclonedds/install"
    ```
    doesnt work.
    
- Python Simulator has not worked yet because `colcon build` had some unsolved problems (may relate to the current virtual environment), hence the C++ simulation was used instead
 
## Teleop
- Required packages:
  ```bash
    pip install scipy
  ```
- Run the ROS2 node `sensorsuit_node.py` provided in this repo on your computer for correct frequency (sampling time) setting.
- Change the `SENSORSUIT_SERVER_ADDRESS` parameter if you use the `sensorsuit_node.py` file in this repo.

## Run
- Run the controller in  the RasPi
- Run the `sensorsuit_node.py` ROS2 node:
  ```bash
  python3 sensorsuit_node.py
  ```
- Put the puppet in T-pose and run the teleop file:
  ```bash
  python3 teleop_angle.py
  ```
- Open C++ Simulator
  ```bash
  cd unitree_mujoco/simulate/build/
  ./unitree_mujoco simulation
  ```
