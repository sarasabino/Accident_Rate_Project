# Accident Rate Project

An incidence rate (also known as an accident rate) is a way of measuring the safety of a workplace. The rate is calculated based on the number of accidents that happen during a particular period of time.
The Occupational Safety and Health Agency (OSHA), the federal agency that establishes safety regulations for workplaces, has a formula for calculating incidence rates. To determine the accident rate in a workplace, start by multiplying the number of accidents by 200,000. Then divide the result by the number of labor hours.

Accidents directly impact two crucial factors for business: money and reputation. Companies have been trying to reduce their injuries rates during the last years, indeed incidence rates have fallen 75% since 1972. Although this good metric there is still a lot of work to do. In 2019, employees in the US suffered 2.8 million workplace injuries. 

This project aims to help reduce those numbers by predicting if an employee is going to suffer an accident/injury. This prediction will help companies to avoid those dangerous situations by being more aware of them. 

As well as predicting the accidents/no accident of the employees on our company, an analysis has been made to identify the company structure, workers profile, location etc...

## Author ✒️

Sara Sabino Martínez

## Stating point & random data generation

## Analysis conclusions

## Modeling

### Feature understanding and selection

need to oversample data

### Seeking best Machine Learning model

We are going to try multiple machine learning models and see which one has better metrics when predicting if an employee is going to have an accident or not. Below we are going to detail the models and its parameters as well as its results.

#### Logistic Regression

- accuracy: 0.89
- precision "No" : 0.80
- precision 'Yes': 0

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/logistic_regression_cm.PNG)


#### K Nearest Neighbors

- accuracy: 0.80
- precision 'No' : 0.84
- precision 'Yes': 0.49

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/neighbors_cm.PNG)

#### Decision Tree

In order to achieve better results we did a GridSearchCV to find the best hiperparameters for our model. 
To solve our problem with the minority class 'Yes', we tried to assign proportionally calculated weights to the model but it delivered bad results as well. The second option that we tried was to oversample the minority class to balance our classes and achieve better results.

First results achieved: accuracy of 0.8 but with a 0 for the 'Yes' class.
![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/first_decision_tree_cm.PNG)

Latest results obtained with the oversample technique and hiperparameter optimization:

- accuracy: 0.76
- precision 'No' : 0.79
- precision 'Yes' : 0.74

Althought the total accuracy of the model has decreased we achieved to got almost the same accuracy for both classes.

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/decision_tree_cm.PNG)

#### Random Forest Algorithm

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/random_forest_cm.PNG)
![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/rforest_learning_curve.PNG)

#### XGBoost

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/xgboost_cm.PNG)

#### SVM: Support Vector Machine

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/svm_cm.PNG)

### Applying Deep Learning: Neural Networks

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/neuraln_loss.PNG)

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/neuraln_accuracy.PNG)

## Conclusions

## Front-end 🚀
https://share.streamlit.io/sarasabino/accident_rate_project/main/Src/Notebooks/05_streamlit_app.py


## Built with 🛠️

* Python
* Streamlit


⌨️ with ❤️ by [sarasabino](https://github.com/sarasabino) 😊
