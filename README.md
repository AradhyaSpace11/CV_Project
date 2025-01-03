# Automatic Highway Emergency Braking System

A real-time road object detection system designed to enhance highway safety by monitoring the middle portion of the video feed and triggering a braking alert if the total detected area of objects increases rapidly. This is especially useful for preventing accidents when the car ahead brakes suddenly at night.

## Installation and Usage

Follow these steps to set up and run the project:

```bash
# Clone the repository
git clone https://github.com/AradhyaSpace11/CV_Project

# Navigate into the project directory
cd CV_Project

# Install the required dependencies
pip install -r requirements.txt

# Run the detection script
python detcarandtruck.py
```

## Features
- **Middle-Region Detection**: Focuses detection on the middle portion of the video feed, ensuring relevant road objects are prioritized.
- **Braking Alerts**: Triggers an alert if a rapid increase in object area is detected, simulating a real-time braking system.
- **Highway Safety**: Designed to assist drivers by providing an early warning for potential hazards, especially useful in low-visibility conditions like nighttime driving.

## How It Works
The system uses a YOLOv8 model to detect road objects such as cars, trucks, and pedestrians within the central third of the video frame. It calculates the cumulative area of detected objects over the last 10 frames and compares it to identify rapid increases. If such an increase is detected, it simulates a braking alert, which can serve as a safety enhancement for autonomous or assisted driving systems.

### Example Use Case
Imagine driving on a highway at night. If the car in front suddenly brakes, the system detects the increased proximity of the vehicle and triggers an alert. This functionality provides critical reaction time for the driver, reducing the likelihood of collisions.
