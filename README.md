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

## Testing with Real-Life Brake-Checking Video
If you want to test the system on a brake-checking real-life video, follow these steps:

1. A sample video, `video.mp4`, is provided in the `videos` folder.
2. To perform inference on this video, run the following command:

```bash
python vid.py
```

3. If you have any other videos to test, place the video files in the `videos` folder and modify the `video_path` variable in `vid.py` as follows:

```python
video_path = 'videos/your_video_name.mp4'
```

Replace `your_video_name.mp4` with the name of your video file.

## How It Works
The system uses a YOLOv11 model to detect road objects such as cars, trucks, and pedestrians within the central third of the video frame. It calculates the cumulative area of detected objects over the last 10 frames and compares it to identify rapid increases. If such an increase is detected, it simulates a braking alert, which can serve as a safety enhancement for autonomous or assisted driving systems.

### Frame Logic and Trigger Mechanism
The braking system is based on frame logic that monitors the bounding box areas of detected objects over time. Here is how it works:

1. **Area Calculation**:
   For each frame, the system calculates the bounding box area for each detected object (car or truck). The areas are summed to get the total area of detected objects in the current frame.

2. **Historical Average**:
   The system maintains a rolling history of total areas from the last 10 frames. The historical average is used as a baseline for comparison.

3. **Trigger Condition**:
   If the ratio of the total area in the current frame to the historical average exceeds a predefined threshold (e.g., 1.50), the system identifies it as a rapid increase in object proximity. This triggers a braking alert.

#### Frame Logic Equation
The frame logic can be expressed as:

$$
\text{Area Ratio} = \frac{\text{Current Total Area}}{\text{Historical Average Area}}
$$

The trigger condition is:

$$
\text{Area Ratio} > \text{Threshold (e.g., 1.50)}
$$

If the above condition is met, the system simulates braking by displaying an alert and blocking further triggers for a predefined braking duration (e.g., 3 seconds).

### Example Use Case
Imagine driving on a highway at night. If the car in front suddenly brakes, the system detects the increased proximity of the vehicle and triggers an alert. This functionality provides critical reaction time for the driver, reducing the likelihood of collisions.
