# Data-Analysis-and-Machine-Learning-Projects
Projects on Machine Learning.
Used Python, Jupyter Notebook, and other Python libraries.

## This Repo Projects are:

## 1. Classification of Iris Flower
#### This is the first project which I have been statrted learning machine learning. I find it really exciting and enjoying. I have tried several models on the dataset. Logistic regression, KNN, Decision Trees, Naive Bayes, Random forest, etc

## 2. Classification of Breast Cancer
#### This is a practice project which I have been taken dataset from UCI Repo and trained model using classification algorithms to predict the result like tumour is benign or malignant.

## 3. Credit Card Fraud Detection
#### Project Involves in detecting frauds in credit card. Major intention behind solving this problem is Annual global fraud losses reached $21.8 billion in 2015, according to Nilson Report.Probably you feel very lucky if you are a fraud. About every 12 cents per $100 were stolen in the US during the same year.In this part of the series, we will train an Autoencoder Neural Network (implemented in Keras) in unsupervised (or semi-supervised) fashion for Anomaly Detection in credit card transaction data. The trained model will be evaluated on pre-labeled and anonymized dataset.
 #### We will be using Tensorflow 1.2 and Keras 2.04 in this project.
 #### About Dataset: All variables in the dataset are numerical. The data has been transformed using PCA transformation(s) due to privacy reasons. The two features that haven't been changed are Time and Amount. Time contains the seconds elapsed between each transaction and the first transaction in the dataset.
#### Autoencoders: Autoencoders can seem quite bizarre at first. The job of those models is to predict the input, given that same input.Definitely was for me, the first time I heard it.More specifically, let’s take a look at Autoencoder Neural Networks. This autoencoder tries to learn to approximate the following identity function: $$\textstyle f_{W,b}(x) \approx x$$ While trying to do just that might sound trivial at first, it is important to note that we want to learn a compressed representation of the data, thus find structure. This can be done by limiting the number of hidden units in the model. Those kind of autoencoders are called undercomplete.
#### Reconstructing Errors: Reconstruction errorWe optimize the parameters of our Autoencoder model in such way that a special kind of error - reconstruction error is minimized.

![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Credit%20Card%20fraud%20detection%20using%20deep%20learning/Screen%20Shot%202020-09-18%20at%203.05.06%20PM.png?raw=true "Title")

#### The ROC curve plots the true positive rate versus the false positive rate, over different threshold values. Basically, we want the blue line to be as close as possible to the upper left corner. While our results look pretty good, we have to keep in mind of the nature of our dataset. ROC doesn't look very useful for us.

### Precison VS Recall

![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Credit%20Card%20fraud%20detection%20using%20deep%20learning/Screen%20Shot%202020-09-18%20at%203.06.49%20PM.png)

![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Credit%20Card%20fraud%20detection%20using%20deep%20learning/Screen%20Shot%202020-09-18%20at%203.07.01%20PM.png)

![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Credit%20Card%20fraud%20detection%20using%20deep%20learning/Screen%20Shot%202020-09-18%20at%203.07.12%20PM.png)
#### Precision measures the relevancy of obtained results. Recall, on the other hand, measures how many relevant results are returned. Both values can take values between 0 and 1. You would love to have a system with both values being equal to 1.

### Prediction:
#### Our model is a bit different this time. It doesn't know how to predict new values. But we don't need that. In order to predict whether or not a new/unseen transaction is normal or fraudulent, we'll calculate the reconstruction error from the transaction data itself. If the error is larger than a predefined threshold, we'll mark it as a fraud (since our model should have a low error on normal transactions).
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Credit%20Card%20fraud%20detection%20using%20deep%20learning/Screen%20Shot%202020-09-18%20at%203.07.26%20PM.png)

### Conclusion:
#### We've created a Deep Autoencoder in Keras that can reconstruct what non fraudulent transactions looks like. Think about it, we gave a lot of one-class examples (normal transactions) to a model and it learned (somewhat) how to discriminate whether or not new examples belong to that same class. Our dataset was kind of magical, though. We really don't know what the original features look like.

### References
#### Building Autoencoders in Keras
#### Stanford tutorial on Autoencoders
#### Stacked Autoencoders in TensorFlow



## 4. Facial Expression Recognition

#### This project involves in recognzing the facial expressions using keras and tensorflow. I used keras 2.3.0 and Tensorflow 2.0.1.
#### I used sequential model to build neural network. Optimizer is adam optimizer.
#### Loaded the image dataset consisting of different emotions like happy, surprise, neutral, angry, sad, disgust and fear.  Each emotions dataset contained more than 3000 images excpet disgust which has only 436 images to train.

![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Facial%20Expression%20Recognition%20with%20Keras/Screen%20Shot%202020-09-18%20at%204.25.58%20PM.png)

#### Later generated training and validation batches with batch size of 64. And converted image pixels to vector. to feed into model
#### Created CNN model with 4 convolution layers with relu activations and with dropout value of 0.25, one flatten layer And 2 fully connected layers.
#### Feed the model with input shape of (48,48,1)
#### The hyperparameters are learning rate==0.0005, loss='Categorical_crossentropy'
#### The result of the model achieved as training loss = 0.8904 - aaacuuracy = 0.6672- val_loss = 0.9813 - val_accuracy = 0.6353

![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Facial%20Expression%20Recognition%20with%20Keras/Screen%20Shot%202020-09-18%20at%204.25.39%20PM.png)

#### Later used this model test on video clip on web browser.


## Human Activity Recognition with smartphone
#### This problem involves in recognizing activities using a smartphone device. This dataset is obtained from UCI machine learning repository.
#### This dataset contains infomation about an experiment carried with group of 30 volunteers wuthuin age of 19-48 years.
#### Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.
#### Based on the given dataset our model has to predict what kind of activity is done by those people based on the values.
#### This is general classifier probelm which we need to classify the each activity based on given set of values so here we are using classification algorithms like randome forest classsifier, logistic regression, decisio tree classifier,  KNN Classifier,  and Gaussian Naive Bayes.
#### Found the highly correlatee and least correalted values and took the absolute correlation which is greater than 0.8 to feed into model
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/HAR%20Using%20smartphone/Screen%20Shot%202020-09-18%20at%204.45.52%20PM.png)

#### Abslotute correlation
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/HAR%20Using%20smartphone/Screen%20Shot%202020-09-18%20at%204.46.16%20PM.png)


#### Split the train and test set to feed model and fed the processed data to above mentioned classifier model to obtain results.

#### Here our logistic regression model made the best prediction than other models
#### Confusion Matrics 
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/HAR%20Using%20smartphone/Screen%20Shot%202020-09-18%20at%204.46.37%20PM.png)


Keras gave us very clean and easy to use API to build a non-trivial Deep Autoencoder. You can search for TensorFlow implementations and see for yourself how much boilerplate you need in order to train one.

These projects are mainly focussed on the Machine Learning, deep learning and data science fields.
ope for pull request. Contributions can be appreciated!!!
I am ready to work in groups in these fields!!!
