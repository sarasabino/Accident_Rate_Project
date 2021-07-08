# Accident Rate Project
### Author ✒️: Sara Sabino Martínez

An incidence rate (also known as an accident rate) is a way of measuring the safety of a workplace. The rate is calculated based on the number of accidents that happen during a particular period of time.

The Occupational Safety and Health Agency (OSHA), the federal agency that establishes safety regulations for workplaces, has a formula for calculating incidence rates. To determine the accident rate in a workplace, start by multiplying the number of accidents by 200,000. Then divide the result by the number of labor hours.

Accidents directly impact two crucial factors for business: money and reputation. Companies have been trying to reduce their injuries rates during the last years, indeed incidence rates have fallen 75% since 1972. Although this good metric there is still a lot of work to do. In 2019, employees in the US suffered 2.8 million workplace injuries, in Spain 1.3 million accidents occured during 2019. 

This project aims to help reduce those  accidents numbers by predicting if an employee is going to suffer one. This prediction will help companies to avoid those dangerous situations by being more aware of them and which factors influence more. 

As well as predicting the accidents/no accident of the employees on our company, an analysis has been made to identify the company structure, workers profile, location etc...

## Repo Structure
```

├── Data                                          # Data used and generated during this study
│   ├── HS_Accidentabilidad.csv                         # Initial file with the accident's data of last years
│   ├── G_Plantas y Tech.csv                            # This file contains the relation of the plants with their country & technology
│   ├── G_Plantas y Tech_streamlit.csv                  # This data is only used in streamlit and contains the coordinates of the fabrics
│   ├── Datos_plantilla.xlsx                            # Aggreagated staff data for analysis purposes
│   ├── Total_staff_by_employee.csv                     # Data generated in notebook 00
│   ├── random_forest_model                             # Saved Random Forest Model
│   └──staff_encoded.csv                                # Data generated in notebook 03 with coded columns
|
├── Images                                        # Images shown in this README
|
├── src / Notebooks                               # Code
│   ├── 00_Data_generation.ipynb                          # Generation of the random data used in the models 
│   ├── 01_Initial_Analysis.ipynb                       # Data Analysis of the data received in first instance
│   ├── 02_Exploratory_Staff_data.ipynb                   # Data Analysis of employee level
│   ├── 02_1_Exploratory_final_data.ipynb                 # Data Analysis of the random data generated for the study (this notebook contains altair charts not show in github)
│   ├── 03_Correlation_analysis_&_feature_selection.ipynb # Analysis of features and selection
│   ├── 04_Model_selection.ipynb                          # Training and evaluation of all the models studied
│   ├── 05_streamlit_app.py                               # Streamlit app py code
│   └── requirements.txt                                  # Requirements file to run the streamlit app
|
└── README.md
    
```

## Objective

## Conclusions

	Model	Accuracy
0	Logistic Regression	0.680605
1	K Neighbors	0.703915
2	Decision Tree	0.750178
3	Random Forest	0.733185
4	XGBoost	0.750178
5	SVM	0.707000
6	Soft-votting	0.739324
7	Hard-votting	0.738612
8	Bagging Classifier	0.731415
9	Neuronal Network	0.896143

## Link to Front-end App 🚀

https://share.streamlit.io/sarasabino/accident_rate_project/main/Src/Notebooks/05_streamlit_app.py


## Built with 🛠️

* Python
* Streamlit

⌨️ with ❤️ by [sarasabino](https://github.com/sarasabino) 😊
