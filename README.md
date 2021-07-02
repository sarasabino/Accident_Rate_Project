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

In order to compare the results between models we are going to analyze the following metrics:

- Accuracy: is the number of correctly predicted data points out of all the data points.
- Precision: is the ratio of correctly predicted positive observations to the total predicted positive observations.

Both metrics are important for our models as we need to be sure that we have a good precision in both classe, it is crucial to identify those 'yes' classes.

#### Logistic Regression

At first it seems this results were good because the accuracy of the model is 0.89, but if we focus on the performance with the 'Yes' label is 0, so the model is always predicting 'No'. We need to improve that prediction by balancing the dataset and running again the model.

- accuracy: 0.80
- precision "No" : 0.80
- precision 'Yes': 0

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/logistic_regression_cm.PNG)

As this results are not valid because it is very important for us to predict when an accident is going to happen, we are going to perform the model again but with the balanced dataset this time.

- accuracy: 0.64
- precision 'No': 0.76
- precision 'Yes': 0.60

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/logistic_regression_cm_os.png)

#### K Nearest Neighbors

- accuracy: 0.80
- precision 'No' : 0.84
- precision 'Yes': 0.49

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/neighbors_cm.PNG)

Although the previous results, were not bad with the unbalanced dataset we are going to try with the balanced one and see if there is any improvement.

-accuracy: 0.74
- precision 1/0: 0.74

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/neighbors_cm_os.png)


#### Decision Tree

In order to achieve better results we did a GridSearchCV to find the best hiperparameters for our model. 
To solve our problem with the minority class 'Yes', we tried to assign proportionally calculated weights to the model but it delivered bad results as well. The second option that we tried was to oversample the minority class to balance our classes and achieve better results.

First result obtained:
- accuracy : 0.82
![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/first_decision_tree_cm.PNG)

Applying the best parameters by the GridSearchCV and without the balanced dataset: accuracy of 0.8 but with a 0 for the 'Yes' class.
- accuracy : 0.80

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/decision_tree_second.png)

Latest results obtained with the oversample technique and hiperparameter optimization:

- accuracy: 0.77
- precision 'No' : 0.81
- precision 'Yes' : 0.74

Althought the total accuracy of the model has decreased we achieved to got almost the same accuracy for both classes.

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/decision_tree_cm.PNG)

#### Random Forest Algorithm

As we have seen that the models need to be balanced, we have performed the random forest with the balanced data from previous steps.

We have also performed a GridSearchCv to try get the best results we can get, although we couldnt increase the first results obtained.

In this model we achieved the following results:

- accuracy: 0.76
- precision 'No' : 0.79
- precision 'Yes' : 0.74

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/random_forest_cm.PNG)
![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/rforest_learning_curve.PNG)

#### XGBoost

We applied a GridSearchCV to find the best parameters, the results obtained were the following ones:

- accuracy : 0.77
- precision 'No' : 0.81
- precision 'Yes' : 0.74

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/xgboost_cm.PNG)

#### SVM: Support Vector Machine
As SVM works well with small datasets and not awesome with large ones and it will take forever to optimize with cross validations, we downsample both categories to see how it will perform. 
We reduced both labels to 2000, having a total dataset of 4000.
The first time we performed this model we achieved the following results:
- accuracy: 0.69
- precision 'No': 0.75
- precision 'Yes': 0.65

Performing the GridSearchCV to find the best parameters we have increase a little bit the model's metrics.

- accuracy: 0.71
- precision 'No': 0.75
- precision 'Yes': 0.67

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/svm_cm.PNG)

### Applying Deep Learning: Neural Networks

In order to evaluate the neural networks models we are going to perform the following metrics:

- Categorical accuracy: is the number of correctly predicted data points out of all the data points.
- Loss: determine how far the predicted values deviate from the actual values in the trainning data.

First model use:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 32)                512       
_________________________________________________________________
dense_1 (Dense)              (None, 16)                528       
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 34        
=================================================================
Total params: 1,074
Trainable params: 1,074
Non-trainable params: 0
```

Before normalizing the data the neural network delivered the following results:

- accuracy: 0.92
- loss: 0.69


After normalizing the data and use the reduced dataset, we create the following model:
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 32)                416       
_________________________________________________________________
dense_4 (Dense)              (None, 16)                528       
_________________________________________________________________
dense_5 (Dense)              (None, 2)                 34        
=================================================================
Total params: 978
Trainable params: 978
Non-trainable params: 0
```


In the first and second layer we used 'relu' as activation function and in the third we used 'softmax'.
The results obtained the first time with 30 epochs and a batch_size of 10:

- accuracy: 0.88
- loss: 0.40

The second time we run this model we did it with 64 epochs and a batch_size of 40. We obtained the following results:
- Categorical accuracy: 0.92
- Loss: 0.40

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/neuraln_loss.PNG)

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/neuraln_accuracy.PNG)

## Conclusions

## Front-end 🚀
https://share.streamlit.io/sarasabino/accident_rate_project/main/Src/Notebooks/05_streamlit_app.py


## Built with 🛠️

* Python
* Streamlit


⌨️ with ❤️ by [sarasabino](https://github.com/sarasabino) 😊
