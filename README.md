# Easyfid
# Face Recognition System

## Overview
This project implements a face recognition system in Python using OpenCV. It allows users to capture facial images, train a recognizer on these images, and then use the trained model to recognize faces in real-time. 

## Features
- Capture multiple facial images per user.
- Train a Local Binary Patterns Histograms (LBPH) face recognizer.
- Real-time face recognition with confidence scoring.

## Requirements
- Python 3.x
- OpenCV (`cv2`) library
- NumPy
- Pillow (PIL)

## Installation
- Clone the repository to your local machine:
    git clone https://github.com/nachovoss/easyfid.git

- Install the required packages:
 ```
    pip install numpy opencv-python pillow
```
    or
    ```
    pip install -r requirements.py
    ```


## Usage
To use the face recognition system, follow these steps:

1. Run the main script:
    ```python
    python Example.py
    ```

2. Follow the prompts to capture images, train the recognizer, and start the face recognition process:
    - Enter the number of pictures per person.
    - Enter the number of users for creating the dataset.
    - Train the recognizer.
    - Enter the number of users for recognition and their names.

## Contributing
Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## License
[MIT License](LICENSE)
