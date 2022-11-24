import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

data = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

#delete columns that have no use
data = data.drop([ 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1)

#encode categorical data
data['Sex'] = data['Sex'].replace('male', 0)
data['Sex'] = data['Sex'].replace('female', 1)

data['Embarked'] = data['Embarked'].replace('C', int(0))
data['Embarked'] = data['Embarked'].replace('Q', int(1))
data['Embarked'] = data['Embarked'].replace('S', int(2))

#fill null values with mean for age
data['Age'].fillna(round(data['Age'].mean()),inplace=True)

#drop null values for embarked
data.dropna(subset=['Embarked'],inplace=True)


titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["PassengerId","Pclass","Sex","Age","SibSp","Parch","Embarked"], 
    description="clean titanic dataset")
titanic_fg.insert(data, write_options={"wait_for_job" : False})

#expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="iris_dimensions")    
#value_between(expectation_suite, "sepal_length", 4.5, 8.0)
#value_between(expectation_suite, "sepal_width", 2.1, 4.5)
#value_between(expectation_suite, "petal_length", 1.2, 7)
#value_between(expectation_suite, "petal_width", 0.2, 2.5)
#iris_fg.save_expectation_suite(expectation_suite=expectation_suite, validation_ingestion_policy="STRICT")    
    

