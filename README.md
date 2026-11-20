# Driver Monitoring System (DMS)

A real-time computer vision system for detecting driver fatigue, drowsiness, and distraction to enhance road safety.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

## Features

- **Drowsiness Detection**: Eye Aspect Ratio (EAR) based eye closure monitoring
- **Yawn Detection**: Mouth Aspect Ratio (MAR) based yawn tracking
- **Distraction Detection**: Head pose estimation using solvePnP
- **Real-time Alerts**: Audio beeps and voice warnings
- **Visual Dashboard**: Live metrics overlay with progress bars
- **Event Logging**: Session statistics saved to JSON/CSV

## Project Structure

```
Driver Monitoring System/
├── src/
│   ├── detector/         # Face, eye, mouth, head pose detection
│   ├── alerts/           # Audio and visual alert systems
│   ├── classifier/       # Future: ML classifiers
│   ├── dashboard/        # Future: Streamlit dashboard
│   └── utils/            # Config and logging
├── models/               # Pre-trained models
├── data/                 # Test videos and images
├── logs/                 # Session logs
├── tests/                # Unit tests
├── main.py               # Entry point
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with webcam (default)
python main.py

# Run with video file
python main.py --video path/to/video.mp4

# Enable debug visualization
python main.py --debug

# Disable audio alerts
python main.py --no-audio
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `d` | Toggle debug mode |
| `r` | Reset counters |

## Detection Thresholds

Configurable in `src/utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| EAR_THRESHOLD | 0.25 | Eye closure threshold |
| MAR_THRESHOLD | 0.6 | Yawn detection threshold |
| YAW_THRESHOLD | 30° | Head turn threshold |
| PITCH_THRESHOLD | 20° | Head tilt threshold |

## Technical Details

### Eye Aspect Ratio (EAR)
```
EAR = (|P2-P6| + |P3-P5|) / (2 × |P1-P4|)
```
- Open eye: EAR ≈ 0.3
- Closed eye: EAR < 0.2

### Mouth Aspect Ratio (MAR)
```
MAR = vertical_distance / horizontal_distance
```
- Normal: MAR < 0.3
- Yawning: MAR > 0.6

## Sample Output

The system displays:
- Real-time metrics panel (EAR, MAR, head angles)
- Progress bars for fatigue indicators
- Alert banners for detected events
- Face bounding box and landmark visualization (debug mode)

## Acknowledgments

- MediaPipe for face mesh detection
- OpenCV for computer vision operations
