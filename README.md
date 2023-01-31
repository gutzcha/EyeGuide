# Introduction
Gaze tracking and gesture recognition technologies have made significant advancements in recent years, with applications ranging from virtual and augmented reality, gaming, and human-computer interaction. This project aims to develop an application that leverages these technologies to allow users to interact with their devices using their gaze and gestures alone.

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

## App Development:

  *  Integrate the gaze tracking and gesture capturing modules into a user-friendly application.
  *  Design a paradigm where users can play a game using their gaze and gestures to collect data and improve model performance.

# Unique Features:

* An open-source and low-cost solution for gaze tracking and gesture recognition.
* Use of computer vision techniques to detect facial landmarks and iris locations.
* Integration of a DL models for facial gesture recognition and gaze tracking.
* Option to add custom gestures through a gesture-recording model using few-shot learning.

# Skills and Requirements to Contribute to the Project:

   * Experience with computer vision, machine learning, and deep learning
   * Proficiency in programming languages such as Python and C++
   * Knowledge of face detection and recognition algorithms
   * Familiarity with libraries such as MediaPipe, CVzone and Pytorch
   * Ability to work with temporal (time series) data
   * Good understanding of neural networks and training process
   * Ability to work with large datasets and extract insights from data
   * Familiarity with Git and GitHub
   * Familiarity with agile development methodologies
   * Understanding of ethical and privacy concerns in AI and computer vision

## For Junior Developers and Data Scientists:

   * Strong interest in learning and experience in computer vision and machine learning
   * Willingness to work in a team and learn from senior developers
   * Ability to communicate effectively and work well with others
   * Ability to work with deadlines and prioritize tasks
   * Good problem-solving skills

## For Designers:

   * Strong skills in user interface (UI) and user experience (UX) design
   * Understanding of the importance of accessibility and user-centered design
   * Ability to create visually appealing designs that are easy to use
   * Ability to work with developers to implement designs in code
   * Familiarity with design tools such as Sketch and Figma

## For Project Managers:

   * Ability to lead a team of developers and other contributors
   * Good organizational skills and attention to detail
   * Ability to prioritize tasks and work with deadlines
   * Ability to communicate effectively with team members and stakeholders
   * Familiarity with agile development methodologies and project management tools such as JIRA and Trello.

## For UI/UX Developers:

   * Strong skills in front-end development
   * Ability to create user-centered interfaces that are accessible and easy to use
   * Familiarity with HTML, CSS, JavaScript, and React
   * Ability to work with designers to implement designs in code
   * Ability to test and debug code to ensure a smooth user experience.


# Resources
The project will require a webcam, a standard desktop or laptop computer, and access to the internet for downloading packages and datasets.

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

# Future Development
In the future, the project may expand to include additional input methods, such as voice commands, as well as additional output methods, such as haptic feedback.
