# Mujoco Simulation
- Follow instruction from link: https://github.com/unitreerobotics/unitree_mujoco
- Note: The interface parameter in unitree_mujoco/simulate/config.yaml must be set correctly (the same as the interface in CYCLONEDDS_URI)
- Mujoco version for cpp: 3.3.6
- If 2. step in the installation for C++ not successful with error error: "‘uint8_t’ does not name a type":
  Add "#include <cstdint>" to "X/unitree_mujoco/simulate/src/joystick/jstest.cc" and compile again (X is the root directory of unitree_mujoco)
  
- Python Simulation: 
    Possible error: "Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH" --> Follow solution provided, try also "export CYCLONEDDS_HOME=~/cyclonedds/install" (No "") if 'export CYCLONEDDS_HOME="~/cyclonedds/install"' (with "") doesnt work.
    
- Python Version did not work because colcon build had problem --> use cpp simulation


# unitree_ros2
- To fix the sudo apt update not working problem, /usr/share/keyrings/ros-archive-keyring.gpg was replaced by: 
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null

    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    The old key is stored under /usr/share/keyrings/ros-archive-keyring_old.gpg

- Follow instruction on https://github.com/unitreerobotics/unitree_ros2, replace all ros2 distro name in commands to the correct distro (here jazzy) (e.g. sudo apt install ros-jazzy-rmw-cyclonedds-cpp)

- The following need to be done after the colcon build of cyclonedds_ws:
    1) change all ros2 distro name in the following setup file to the correct distro (here jazzy):
    X/unitree_ros2/setup.sh
    X/unitree_ros2/setup_local.sh
    X/unitree_ros2/setup_default.sh
    
    2) (Optional) To avoid sourcing the above files every time, "source ~/Desktop/quan/unitree_setup.bash" can be added to .bashrc. The wanted setup file can then be chosen every time a new terminal opens.
    
- Errors:
    1) Running the "./unitree_mujoco" simulation inside "unitree_mujoco/simulate/build" can cause crashing problem because of a DDS runtime mismatch. This can be fixed by the line "export LD_LIBRARY_PATH=/opt/unitree_robotics/lib:$LD_LIBRARY_PATH" befor running the simulation. (this line is also added to the unitree_setup.bash file. 
    2) After changing the Network interface name in 
        export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="wlp0s20f3" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
        If CYCLONEDDS_URI seems not to be applied (e.g. topics from a previously connected network can still be seen, or "ros2 topic list" spins forever without output any topics when an unwanted (previously connected) network interface is not in connection), reconnect to the previously connected network or reboot if reconnection fails and restart ros2 daemon by:
                    ros2 daemon stop
                    ros2 daemon start
                    
    
# ROS2 in conda python 3.12
- ROS2 only support python3.12, so a conda environment with python3.12 is needed.
- Required parameter: 
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
- Required installation:
    - pip install empy
    - pip install catkin_pkg
    - pip install lark lark-parser
    - Any ROS2 packages previously built with python 3.13 (conda base) need to be rebuilt with python3.12
    
# Teleop
- Required packages:
    pip install scipy
    
