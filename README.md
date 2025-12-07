Worker Safety Helmet Detection System

This project is a real-time worker safety monitoring platform that utilizes YOLOv8 (You Only Look Once, version 8) for detecting workers and verifying if they are wearing helmets. It is designed to improve safety in workplaces, such as factories and construction sites, by ensuring that workers are complying with safety gear requirements.

Features
1. Person Detection (First Layer)

YOLOv8 is used to detect individuals in a live video stream.

The model is trained to identify the presence of people, even in dynamic environments with multiple workers.

2. Helmet Detection (Second Layer)

After detecting a person, a second layer of the model checks whether the person is wearing a helmet.

This detection is important for enforcing safety regulations in industrial and construction environments.

The system flags workers not wearing helmets in real time.

3. Real-Time Video Processing

The system processes video from a webcam in real time.

As the video feeds in, the model performs detection and overlays the results (e.g., whether a person is wearing a helmet) on the video.

4. Admin Panel

The project features an admin panel that provides a simple user interface for viewing live output from the webcam.

The admin panel also shows the real-time detection results, including details about the worker’s helmet status.

5. Worker Safety Monitoring

This system ensures that workers are wearing helmets, which can significantly reduce the risk of head injuries.

The application is ideal for factories, construction sites, or any workplace where worker safety needs to be monitored actively.

Setup Instructions
1. Clone the Repository

Start by cloning the repository to your local machine:

git clone https://github.com/your-username/helmet-detection.git
cd helmet-detection


This will create a folder called helmet-detection with all the project files.

2. Set Up Virtual Environment

To avoid conflicts between dependencies, it's best to create a virtual environment. You can do this by running the following command:

python -m venv helmet_env


Activate the virtual environment:

On Windows:

.\helmet_env\Scripts\Activate


On macOS/Linux:

source helmet_env/bin/activate


After activation, your terminal should indicate that you're working inside the virtual environment.

3. Install Dependencies

Install the required libraries from the requirements.txt file. This file includes all the dependencies needed to run the project, including FastAPI, YOLOv8, OpenCV, and others.

pip install -r requirements.txt

4. Run the FastAPI Server

Once the dependencies are installed, you can start the FastAPI server. This will serve both the backend logic and the admin panel's user interface.

uvicorn main:app --reload


This will start the FastAPI server in development mode, and you can access the application in your web browser at:

http://127.0.0.1:8000

Project Structure

Here’s a breakdown of the folder structure and the purpose of each file:

helmet-detection/
│
├── main.py           # FastAPI backend for handling webcam video feed, YOLOv8 processing, and API requests
├── models/           # Directory containing YOLOv8 model files (e.g., `.pt` or `.onnx` models for person and helmet detection)
├── templates/        # Folder containing HTML templates for the admin panel interface
├── static/           # Folder for static files (CSS, JavaScript) to style the admin panel and control frontend
├── requirements.txt  # List of Python libraries and dependencies required to run the project
├── README.md         # This documentation file explaining the project
└── LICENSE           # License file (e.g., MIT License or any other applicable license)

Technologies Used
1. YOLOv8 (You Only Look Once)

YOLOv8 is a state-of-the-art object detection model capable of detecting multiple objects in an image or video stream in real time.

The first layer of YOLOv8 is trained to detect people, and the second layer detects whether those people are wearing helmets.

YOLOv8 is fast and efficient, making it ideal for real-time applications like this one.

2. FastAPI

FastAPI is a modern web framework for building APIs with Python. It's known for its speed, simplicity, and automatic generation of OpenAPI documentation.

FastAPI is used to serve the backend, handle the detection requests, and power the admin panel interface.

3. OpenCV

OpenCV is a powerful library used for computer vision tasks. It is used here to capture webcam video, process frames in real-time, and display the results.

OpenCV allows the YOLOv8 model to analyze the video feed and overlay the results (helmet detection) onto the video.

4. HTML/CSS/JavaScript

The admin panel is built with a simple combination of HTML, CSS, and JavaScript.

This provides a user-friendly interface where administrators can view the real-time video stream from the webcam, along with the detection results (e.g., whether the worker is wearing a helmet).

Future Improvements

This project can be expanded with additional features:

Multi-Object Detection

Improve the accuracy and performance of detecting multiple workers in crowded environments, which is common in workplaces like factories.

Alerting and Notification System

Integrate email, SMS, or push notifications to alert managers or administrators when a worker is detected without a helmet.

Database Integration

Store historical data of detected workers and their helmet usage for reporting and analytics.

Implement a database system (like SQLite or PostgreSQL) to log detection events and worker information.

Mobile Application Integration

Create a mobile application that allows supervisors to monitor helmet usage on-site from their smartphones or tablets.

Implement real-time notifications and streaming for easier remote management.

Enhanced Detection Models

Improve the accuracy of the helmet detection by training the model on more diverse datasets and using techniques like transfer learning.

License

This project is licensed under the MIT License - see the LICENSE
 file for more information.
