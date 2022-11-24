# scalable-serverless-machine-learning-system

Aim –
This lab contains primarily two tasks. The first task corresponds to building and running the Iris Flower Dataset as a serverless system. The second task corresponds to building a similar serverless ML service for the Titanic passenger survival dataset.





Datasets – 
Iris dataset:
The Iris dataset contains four features – the length and width of sepals and the length and width of the petals. There are three labels - Iris setosa, Iris virginica and Iris versicolor.



Titanic dataset:
The Titanic dataset contains twelve features – PassengerID, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin and	Embarked. These features are used to predict if the passenger survived or not. 


Steps – 
titanic-feature-pipeline.py/iris-feature-pipeline.py 
These files deal with reading the dataset from GitHub links, performing data preprocessing, and uploading the data into a feature group on Hopsworks. 
Feature engineering for the titanic dataset: The columns PassengerId, Name, Ticket, Cabin and Fare were removed considering their low predictive power. Male was encoded as 0 while female was encoded as 1. In the ‘Embarked’ column, C, Q and S were encoded as 0,1 and 2 respectively. The missing values in the ‘age’ column were filled by the mean age while the rows containing missing values in the ‘Embarked’ column were removed. 


titanic-training-pipeline.py/iris-feature-pipeline.py
A feature view is created from the feature group. This represents the training schema. The dataset is split into training and testing (80% for training and 20% for testing). 
A K-nearest neighbor with two neighbors is used for training the Iris dataset while a Random Forest Classifier with 100 estimators is used to train the titanic dataset. 
The classification report, the confusion matrix and the model are uploaded to Hopsworks. 


titanic-feature-pipeline-daily.py/iris-feature-pipeline-daily.py
Randomly generated synthesized data is added to the feature group on Hopsworks once per day. For the iris dataset, random values (within a specified range) corresponding to one of the flowers is added to the dataset as a single dataframe row. For the titanic dataset, random values are generated for each feature and the label is marked as either survived or didn’t survive. This is then added to the feature group on Hopsworks.

titanic-batch-inference-pipeline.py/iris-batch-inference-pipeline.py
The batch inference pipeline is used to predict the labels of the batch data once per day. The predicted and actual values of the data added by the feature-pipeline-daily.py file is displayed. It also displays the confusion matrix along with the previous record of predictions. 



UI –

Predicting label by entering the data:

Iris: https://huggingface.co/spaces/Vasi001/iris

Titanic: https://huggingface.co/spaces/Vasi001/titanic

Monitoring:

Iris: https://huggingface.co/spaces/Vasi001/iris-monitor

Titanic: https://huggingface.co/spaces/Vasi001/titanic-monitor
