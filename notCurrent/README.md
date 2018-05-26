# Project Title

The aim of this project is to use sensors registering Electromyography signals in order to be able to classify functional hand gestures. This will be done using a relatively inexpensive armband placed on the forearm of the subject. Firstly, different machine learning techniques will be assessed using NinaPro database: 
* Multi Layer Perceptron Neural Network
* Suppor Vector Machine
* Recurrent Neural Networks

NinaPro is a database created by researchers from around the world to make deep learning gesture recognition projects easier. It contains EMG data both from amputees and intact subjects. After assessing which one of the methods is best suited for this application, the armband will be used to perform live gesture recognition.


## Current progress in the industry and problem statement

Electromyography (EMG) is an electro diagnostic technique for recording signals produced by skeletal muscles. Those signals are produced when the muscle cells are electrically or neurologically activated. Those signals can be captured in two ways:

* intramuscular placement of the electrode
* surface electrode placement

The invasive placement of the electrode inside of the subject's arm is problematic and cannot be used long term. On the other hand the surface electrode placement is prone to high levels of noise in the signals. Testing the outcome provided by the EMG sensors, and following the literature \cite{fat} it became evident that the thickness of the fat layer, where the sensors are placed is an important factor in being able to classify the signals well. Another factor that needs to be assessed is how close to the skin can the sensor be placed and what kind of obstructions are there i.e. hair on the forearm. In EMG medical procedures, a fluid is put between the sensor and the skin, to ensure minimal reflection upon changing the medium. In the normal usage that would be impractical and the disadvantages of not using the fluid will have to be assessed.

Most of the research in the field is concerned with recognising simple gestures like: wrist flexion, wrist extension, supination, pronation, hand open, and hand close. The progress is good and the accuracy is often higher than 95\% , however for reasons mentioned above for some subject the accuracy is much lower. In the literature there seems to be four approaches which seem to give good results:

* Artificial Neural Networks (mostly Multi-layer perceptron) 
* Support Vector Machines 
* Hidden Markov Models 
* Random Forest Regression

The main remaining problem is that those simple gestures are not functional and do not greatly improve amputees quality of life. An algorithm which can classify functional movements like grabbing a glass or a pen is what is needed. With that comes a difficulty of determining the forces needed for those activities. This can be done by implementing a control feedback loop in the prostheses or by further analysing the EMG signals \cite{Forces}.  Nowadays still the vast majority of prostheses are either passive (mainly cosmetic or single purpose) or body-powered (meaning that by means of a harness some basic movement is possible by flexing or tensing a different part of the body).


### Processing Techniques

In order to feed in the NinaPro Dataset and use those methods, the data needs to be pre-processed. Firstly the transitional periods were cut by analysing the time for the signal to stabilise. Next the signals are low-pass filtered at 1Hz using a zero-phase second order Butterworth filter.

Figure below shows the two most important principal components of the data. Each colour represents a different hand posture. It can be noticed that this problem is difficult and creating an algorithm which performs well on all subjects might be very hard. Thus for now the importance will be put on gesture recognition for 1 subject. If time permits the method of calibration for a new subject will be implemented.

![alt Principle Component Analysis of the Dataset](/figures/PCANinaPro.png)



### Multi-Layer perceptron (MLP)

In order to classify the data the first technique to be used was chosen to be a single hidden layer MLP. A neural network code was written from scratch in order to understand the architecture. A standard sigmoid function was used as an activation function and the number of neurons in the hidden layer was optimised. Penalising the synaptic weight between neurons helped with over-fitting. The Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm was used to optimise the network.
