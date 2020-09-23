# Data-Analysis-and-Machine-Learning-Projects
Projects on Machine Learning.
Used Python, Jupyter Notebook, and other Python libraries.

## This Repo Projects are:

## 1. Classification of Iris Flower
#### This is the first project which I have been statrted learning machine learning. I find it really exciting and enjoying. I have tried several models on the dataset. Logistic regression, KNN, Decision Trees, Naive Bayes, Random forest, etc
##### To the data/species distribution in dataset I used pairplot from seaborn library and plotted the data.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Classifiaction%20of%20iris%20flowers/Screen%20Shot%202020-09-23%20at%202.35.09%20AM.png)
##### To know the specific length of sepal and petal of each category I used violin plot to show it.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Classifiaction%20of%20iris%20flowers/Screen%20Shot%202020-09-23%20at%202.36.52%20AM.png)
###### Later to show accuracy of model I used boxplot to display it.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Classifiaction%20of%20iris%20flowers/Screen%20Shot%202020-09-23%20at%202.38.32%20AM.png)

## 2. Classification of Breast Cancer
#### This is a practice project which I have been taken dataset from UCI Repo and trained model using classification algorithms to predict the result like tumour is benign or malignant.
##### Correlation matric showing the relation between dependent variable with features
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Classification%20of%20breast%20cancer/Screen%20Shot%202020-09-23%20at%202.46.14%20AM.png)
##### Showing model accuracy on using different alogirthms on the dataset
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Classification%20of%20breast%20cancer/Screen%20Shot%202020-09-23%20at%202.46.27%20AM.png)
##### displaying accuracy of model predictions
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Classification%20of%20breast%20cancer/Screen%20Shot%202020-09-23%20at%202.46.39%20AM.png)

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
#### Keras gave us very clean and easy to use API to build a non-trivial Deep Autoencoder. You can search for TensorFlow implementations and see for yourself how much boilerplate you need in order to train one.

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


## 5. Human Activity Recognition with smartphone
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

## 6. Hypothesis Test on University Towns

#### A quarter is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
#### A recession is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
#### A recession bottom is the quarter within a recession which had the lowest GDP.
#### A university town is a city which has a high percentage of university students compared to the total population of the city.

### Steps
#### 1) Collected the previous housing prices data from the different website sources like Zillow research data site and University town in United States from wiki.
#### 2) Calculated Recession start, end time and Recession Bottom to know the actual prices.
#### 3)Divided the resulted data Into Quarters and applied Hypothesis test on these Quarters to get the result

## 7. Identifying the target customers using clustering
#### This dataset is taken from UCI machine learning repo this is one of my initial practice problem when I started learning ML. The data set consist of the custumers who spends on buying something based on their salaries. Here my motivation is to identify the target customers which mall owners can target and tend them to buy more from their malls. Here I am using clustering techniques to predict the results. This probelm is simply identifying the target custmer for company to focus on to improve the sales of their product. So here we are using clustering technique to form a clusters of people. 

#### Here we used elbow method to find the optimal number of cluster for our problem. Image is shown below:
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Identifying%20target%20customers%20using%20clustering/Screen%20Shot%202020-09-19%20at%201.37.15%20AM.png)

#### There is another technique to find the optimal number clusters that is using dendrogram. It generally calculate the euclidean distance between the lables.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Identifying%20target%20customers%20using%20clustering/Screen%20Shot%202020-09-19%20at%201.39.16%20AM.png)

#### Here we are using K-Means clustering algortihm to find the solution for our model. We feed our data into this model it will predict the result for us. Here is the result shown
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Identifying%20target%20customers%20using%20clustering/Screen%20Shot%202020-09-19%20at%201.40.56%20AM.png)

#### And we are testing our model using hierarchical clustering also to seee which model predicted with high accuracy. Here is image of Hierachical clusters model predictions
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Identifying%20target%20customers%20using%20clustering/Screen%20Shot%202020-09-19%20at%201.42.42%20AM.png)

## 8. Prediction of Energy Output of power plant
#### A simple ML problem of small dataset which is from UCI Repo. The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant. A combined cycle power plant (CCPP) is composed of gas turbines (GT), steam turbines (ST) and heat recovery steam generators. In a CCPP, the electricity is generated by gas and steam turbines, which are combined in one cycle, and is transferred from one turbine to another. While the Vacuum is colected from and has effect on the Steam Turbine, he other three of the ambient variables effect the GT performance.
#### Predict the energy output of powerplant. Here Our model did very good job with accuracy of 96%
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Prediciton%20of%20EE%20output/Screen%20Shot%202020-09-19%20at%201.59.30%20AM.png)

## 9. Predicting Bike sharing patterns:
#### This project is an real world problem in which Real work data. Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return 
back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic, environmental and health issues.

#### Apart from interesting real world applications of bike sharing systems, the characteristics of data being generated by these systems make them attractive for the research. Opposed to other transport services such as bus or subway, the duration of travel, departure and arrival position is explicitly recorded in these systems. This feature turns bike sharing system intoa virtual sensor network that can be used for sensing mobility in the city. Hence, it is expected that most of importantevents in the city could be detected via monitoring these data.
#### Build model to predict the bike sharing patterns for future months which is basically a regreesion problem.
#### First I did exploratory analysis to find correaltion between features and dependent variable. Plot then using matplotlib.pyplot library.By plotting above we have seen that some labels are highly corelated and some are less corelated for better results we have to rule out the high and low corelated labels and we have to keep only moderatly corealted labels for better accuracy. So we can drop the labels which are high and least corelated to dependent variable.
#### This data is pretty complicated! The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week. Looking at the data above, we also have information about temperature, humidity, and windspeed, all of these likely affecting the number of riders. You'll be trying to capture all this with your model
#### scaled the data built a 3 layered neural network. Tred with several hyperparameters and after tuning got these hyperparamers as best. LR = 0.2, Hidden_nodes = 16, output node = 1, activation = sigmoid.
#### train and validation loss are plotted below.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Predicting%20Bike%20sharing%20Patterns/Screen%20Shot%202020-09-21%20at%202.03.02%20AM.png)
#### Finally our model predicted with accuracy shown in figure.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Predicting%20Bike%20sharing%20Patterns/Screen%20Shot%202020-09-21%20at%202.03.12%20AM.png)

## 10. Sentiment Analysis Using with Deep Learning Using BERT
#### Project Outline

#### Task 1: Introduction
##### Bert is a large scaled transformer based learning model that can be fine-tuned for variety of tasks. We will be using SMILE Twitter dataset You can find it here: Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): SMILE Twitter Emotion dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3187909.v2

#### Task 2: Exploratory Data Analysis and Preprocessing
##### We are her using pytorch for this problem. In this step I imported dataset. Counted the each category or columns of dataset. It contains 5 major emotions i.e happy, non-relavent, angrat, surprise, sad, disgust. Now we will predict which tweet is belongs to which category.

#### Task 3: Training/Validation Split
##### Divided dataset into validation and train sets, seperated 15% of dataset for testing purpose.
 
#### Task 4: Loading Tokenizer and Encoding our Data
##### Imported Berttokenizer from transormers and deployed bert-base-uncase. Encoded the both training and validation sets to feed the model

#### Task 5: Setting up BERT Pretrained Model
##### I imported pre trained bert model from transoformers which is bert sequenceclassification and now setting it. 

#### Task 6: Creating Data Loaders
##### from pytorch I am importing Dataloader, RandomSampler, SequentialSampler. Set up the batch size as 4

#### Task 7: Setting Up Optimizer and Scheduler
##### In this step I imported AdamW optimizer as optimization method. Set up learning rate as 0.0005, and eps= le-8 as hyper parameters.

#### Task 8: Defining our Performance Metrics
##### To know the performance of our model we are using some mtrics to measure it. ffirst one is F1 Score. 

#### Task 9: Creating our Training Loop

#### Task 10: Loading and Evaluating model

## 11. Walmart Sales Forecasting
##### This is one of the kaggle problem where I worked on my first time series problem. It is a multivariate time series problem and in which we have to predict the future sales of the walmart based on the given actual walmart data. It is multivariate because many factors are effecting on the dependent variable change.
##### Impoted dataset and done exploratory analysis on the dataset plot the given data sales in each store.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Wallmart%20Sales%20forecasting%20Project/Screen%20Shot%202020-09-23%20at%202.24.02%20AM.png)
##### Later plot data a barchart to see the sales in each departement
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Wallmart%20Sales%20forecasting%20Project/Screen%20Shot%202020-09-23%20at%202.25.04%20AM.png)
##### To see the correaltion between the all the features with dependent variable here I plotted correlation matrix. It defines darker color for highly correaled variable and lighter for least correlated variables.
![Alt text](https://github.com/starkworld/Data-Analysis-and-Machine-Learning-Projects/blob/master/Wallmart%20Sales%20forecasting%20Project/Screen%20Shot%202020-09-23%20at%202.27.02%20AM.png)
* Here, markdown 1 to 5 are weakly co-realted to weekly sales so we will drop those columns

* Also fuel price is strongly co-realted to weekly sales we will drop that one too.

* As well as we will drop temparature, CPI, Unemployment as they are nothing to do with prediciting result.

##### Later built model by importing regression models from scikit-learn and tested all models to predict which model performed best job to predct sales.
##### Here I used KNN, ExtratreeRegressor, RandomForestRegressor, SVM, Neural Net.
##### trained model through this models and calculated error using absolute mean error.
##### Feed the data as 10 cross folds to better performance and tested our model to predict the results 

These projects are mainly focussed on the Machine Learning, deep learning and data science fields.
ope for pull request. Contributions can be appreciated!!!
I am ready to work in groups in these fields!!!
