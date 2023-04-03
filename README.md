# Introduction
Gaze tracking and gesture recognition technologies have made significant advancements in recent years, with applications ranging from virtual and augmented reality, gaming, and human-computer interaction. This project aims to develop an application that leverages these technologies to allow users to interact with their devices using their gaze and gestures alone.

![logo (2)](https://user-images.githubusercontent.com/35958758/218949409-b82f0f3e-54de-4804-a9ff-f40fa4a4a141.png) ![ezgif-4-6f23d675dc](https://user-images.githubusercontent.com/35958758/229591335-1dcce471-0b73-4328-b62b-cdb4ce568703.gif)



# Problem Statement
Traditional input methods like keyboards, mice, and touchscreens have proven to be effective in daily use, but they may become challenging or even inaccessible in hands-free environments such as virtual and augmented reality. Additionally, not all users, including those with physical disabilities, can effectively use these traditional methods. As a result, there is a pressing need for alternative forms of interaction that are more intuitive, user-friendly, and accessible to a wider range of users, including those with disabilities and those looking for more convenient methods of input in immersive environments.

# Target Users
This project aims to serve individuals in need of more natural, user-friendly, and accessible methods of device interaction, particularly those with physical disabilities. Additionally, it caters to virtual and augmented reality users, and the gaming community seeking a hands-free experience.

# Offered Solutions
The application will leverage computer vision algorithms to estimate the user's gaze direction and recognize their facial gestures. The gaze tracking module will use a webcam to estimate the user's gaze direction on the screen, while the gesture capturing module will recognize facial gestures such as winking, eyebrow-raising, and lip-pursing. In the future, the gesture capturing module may be expanded to allow users to create custom gestures.

# Implementation
The gaze tracking module will be implemented using a temporal dilated convolution neural network (TDCNN) trained on a dataset of face images and gaze coordinates. The gesture capturing module will be implemented using a similar neural network trained on a dataset of facial gestures. The application will be built using the MediaPipe and OpenCV libraries and will run on a standard desktop or laptop computer.
## Gaze Tracking:

   * Utilize the mediapipe and cvzone python packages to detect the face mesh, including the irises.
   * Gather and annotate a dataset of face images with the screen coordinates where the subject is looking.
   * Train a deep learning neural network to predict the gaze location based on face mesh landmarks.

##  Gesture Capturing:

   * Train a deep learning neural network model to classify the action or gesture.
   * Define default gestures, including winking, eyebrow-raising, and single or double, left or right eyebrow raises.
   * Implement a trigger gesture to activate the gesture recognition module.
   * Consider adding a gesture-recording module in the future to support custom gestures using few-shot learning methods.
   
![ezgif-4-b2d3fc074b](https://user-images.githubusercontent.com/35958758/229589682-10b031e9-0b52-44cd-864b-00b35361988f.gif) 

![ezgif-4-b2d3fc074b](https://user-images.githubusercontent.com/35958758/229590866-237008a7-e155-45c2-a915-a6390f303f59.gif)

## App Development:

  *  Integrate the gaze tracking and gesture capturing modules into a user-friendly application.
  *  Design a paradigm where users can play a game using their gaze and gestures to collect data and improve model performance.

# Unique Features:

* An open-source and low-cost solution for gaze tracking and gesture recognition.
* Use of computer vision techniques to detect facial landmarks and iris locations.
* Integration of a DL models for facial gesture recognition and gaze tracking.
* Option to add custom gestures through a gesture-recording model using few-shot learning.

# Skills and Requirements to Contribute to the Project:
## For Junior Developers and Data Scientists:

   * Proficiency in Python programming
   * Experience with computer vision, and deep learning   
   * Good understanding of neural networks and training process
   * Familiarity with Git and GitHub
   * Knowledge of face detection and recognition algorithms - advantage
   * Familiarity with libraries such as MediaPipe, opencv and Pytorch - advantage
   * Ability to work with temporal (time series) data - advantage   
   
## For UI/UX Developers:

   * Strong skills in front-end development
   * Ability to create user-centered interfaces that are accessible and easy to use
   * Familiarity with HTML, CSS, JavaScript, and React
   * Ability to work with designers to implement designs in code
   * Ability to test and debug code to ensure a smooth user experience.

# Resources
## Developments:
  * Workstation to train the models
  * Datasets for training
## Usage: 
  * A webcam
  * A standard desktop or laptop computer
  * Access to the internet for downloading packages and datasets.


# Privacy and Ethical Concerns
The application will collect and process images of the user's face, which may raise privacy and ethical concerns. The project will comply with all applicable privacy laws and regulations, and all data collected will be securely stored and processed. The project will also provide users with clear and concise information about what data is being collected and how it is being used.

# Workflow
1. Gather data of face images and gaze coordinates for training the gaze tracking module.
2. Design and train the gaze tracking module using the MediaPipe and OpenCV libraries.
3. Gather data of facial gestures for training the gesture capturing module.
4. Train the gesture capturing module using a similar neural network.
5. Integrate the gaze tracking and gesture capturing modules into the application.
6. Test the application and make improvements as needed.
7. Release the application to the public and gather feedback.
8. Continuously improve the application based on user feedback.

# Open Tasks
- [ ] **Create and train a model for gesture classification**
  - [x] Generate training data
    - [x] Gather data of face videos for unsupervised model pretraining
    - [x] Build pipeline to extract and process pose data from face videos
  - [x] Create a masked autoencoder (MAE) model with ViT and TDCNN backbone
  - [ ] Create a training and validation pipeline and pretrain the model
    - [ ] Create dataset and dataloader classes
    - [ ] Create trasformation classes including masking
  - [ ] Finetune the model on simple facial gestures
    - [ ] Gather facial gesture videos
    - [ ] Train the model
- [ ] **Create and train a model for eye tracking and gaze estimation**
  - [ ] Generate training data
    - [ ] Gather images of faces and coordinates of where they are looking at on the screen
    - [ ] Build pipeline to extract and process data
  - [ ] Create a model (TDCNN?)
  - [ ] Create a training and validation pipeline and pretrain the model

# Future Development
In the future, the project may expand to include additional input methods, such as voice commands, as well as additional output methods, such as haptic feedback.

# About the Author
I am a Ph.D. candidate in Neuroscience at the University of Haifa, focusing on utilizing computer vision and deep learning to analyze animal social behaviors. I recently concluded a successful one-year internship at Amazon Lab 126, where I developed my programming, research, and deep learning skills. I am passionate about using technology for the betterment of society. I am eager to bring my knowledge and experience in computer vision and deep learning to this.

Ways to contact:
* LinkedIn: https://www.linkedin.com/in/goussha/
* Email: gutzcha@gmail.com
* Git: https://github.com/gutzcha

