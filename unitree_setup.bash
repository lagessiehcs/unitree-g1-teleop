conda activate unitree 
# Only run this in interactive shells
if [[ $- == *i* ]]; then
    # Define root directory (change this one if the root directory of unitree_ros2 is different)
    ROOT_DIR=~/Desktop/quan

    # Run commands with ROOT_DIR
    source /opt/ros/jazzy/setup.bash
    source $ROOT_DIR/unitree_ros2/cyclonedds_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    
    STATE_FILE="$ROOT_DIR/last_dds_uri"
    CYCLONEDDS_URI=$(<"$STATE_FILE") 
    
    echo "Select environment:"
    echo "    1: unitree ros2 environment"
    echo "    2: unitree ros2 simulation environment"
    echo "    3: unitree ros2 real robot and simulation environment"
    echo "Enter: unitree ros2 environment with default interface"
    read -p "Enter choice [1/2/3/Enter]: " choice

    case $choice in
        1)
            echo "Setup unitree ros2 environment for real robot"
            export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="enp3s0" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
            ;;
        2)
            echo "Setup unitree ros2 simulation environment"
            export ROS_DOMAIN_ID=1 # Modify the domain id to match the simulation
            export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="wlp0s20f3" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'

            ;;
         3)
            echo "Setup unitree ros2 real robot and simulation environment"
            export ROS_DOMAIN_ID=0 # Modify the domain id to match the simulation
            export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="enp0s31f6" priority="default" multicast="default" />
                            <NetworkInterface name="wlp0s20f3" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'

            ;;
            
        *)
            echo "Setup unitree ros2 environment with default interface"
            ;;
    esac

    # Read the previously saved value (if it exists)
    if [ -f "$STATE_FILE" ]; then
        LAST_URI=$(<"$STATE_FILE") 
    else
        LAST_URI="unset"
    fi

    # Compare current and last URI
    if [ "$CYCLONEDDS_URI" != "$LAST_URI" ]; then
        echo ""
        echo "CYCLONEDDS_URI changed"
        echo "Restarting ROS2 daemon to apply new DDS configuration..."
    
        ros2 daemon stop
        ros2 daemon start

        echo "ROS2 daemon restarted."

        # Save the new URI for next check
        echo "$CYCLONEDDS_URI" > "$STATE_FILE"
    fi
fi

export LD_LIBRARY_PATH=/opt/unitree_robotics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

source $ROOT_DIR/ros_recording/install/setup.bash 

