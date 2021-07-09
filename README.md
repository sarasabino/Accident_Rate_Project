# Accident Rate Project

DATA SCIENCE MASTER TFM

### Author ✒️: Sara Sabino Martínez

An incidence rate (also known as an accident rate) is a way of measuring the safety of a workplace. The rate is calculated based on the number of accidents that happen during a particular period of time.

The Occupational Safety and Health Agency (OSHA), the federal agency that establishes safety regulations for workplaces, has a formula for calculating incidence rates. To determine the accident rate in a workplace, start by multiplying the number of accidents by 200,000. Then divide the result by the number of labor hours.

Accidents directly impact two crucial factors for business: money and reputation. Companies have been trying to reduce their injuries rates during the last years, indeed incidence rates have fallen 75% since 1972. Although this good metric there is still a lot of work to do. In 2019, employees in the US suffered 2.8 million workplace injuries, in Spain 1.3 million accidents occured during 2019. 

## Objective

The main objetives of this project are in first place to understand and analyze to get conclusion from the data so broad the knowledge of the company and to predict which employees are going to have an accident in order to be able to avoid this situations.
This prediction will help companies to avoid those dangerous situations by being more aware of them and which factors influence more. 

## Instructions and structure

In order to run the data please download the notebooks contained in this repository. Each notebook contains at the top a cell with the code dependencies and version used during its execution.The data needed is in the following shared folder: https://drive.google.com/drive/folders/1pzxQWsVrMSZhSZ3pHYQtUpX33XMRjN9g?usp=sharing

The fron end app is develop in streamlit and the link is shared below.

Above a description of the structure of the repo and the data folder is described:

Repo Structure
```
├── Data 
│   └── G_Plantas y Tech_streamlit.csv    		# This data is only used in streamlit and contains the coordinates of the fabrics
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
├── Memoria_Accident_Rate.pdf
└── README.md
    
```
.
Shared folder structure
```
├── Data                                       	# Data used and generated during this study
│   ├── HS_Accidentabilidad.csv                         # Initial file with the accident's data of last years
│   ├── G_Plantas y Tech.csv                            # This file contains the relation of the plants with their country & technology
│   ├── G_Plantas y Tech_streamlit.csv                  # This data is only used in streamlit and contains the coordinates of the fabrics
│   ├── Datos_plantilla.xlsx                            # Aggreagated staff data for analysis purposes
│   ├── Total_staff_by_employee.csv                     # Data generated in notebook 00
│   ├── random_forest_model                             # Saved Random Forest Model
└── └──staff_encoded.csv                                # Data generated in notebook 03 with coded columns

```


## Conclusions
```

	Model			Accuracy	Prec No		Precc Yes
0	Logistic Regression	0.684875	0.708802	0.665868
1	K Neighbors		0.717794	0.694162	0.747974
2	Decision Tree		0.751335	0.745524	0.757427
3	Random Forest		0.732206	0.74042		0.724535
4	XGBoost			0.751335	0.745524	0.757427
5	SVM			0.707000	0.703704	0.711316
6	Soft-votting		0.736833	0.746299	0.728067
7	Hard-votting		0.735676	0.74496		0.72707
8	Bagging Classifier	0.734703		
9	Neuronal Network	0.908714		
```
## Link to Front-end App 🚀

The front-end of this project is develop in a streamlit application, the code can be found in this notebook. 
The app is though to be for a client in a company that wants to see a analytical dashboard of the accidents in the company and then is going to have the option to complete a form in order to find if a particular employee is goign to have an accident.
The model used in the application is a Random Forest Model.

https://share.streamlit.io/sarasabino/accident_rate_project/main/Src/Notebooks/05_streamlit_app.py

Overview of app dashboard:

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/Dashboard_overiew.PNG)

Overview of the form:

![alt text](https://raw.githubusercontent.com/sarasabino/Accident_Rate_Project/master/Images/Prediction_form_overview.PNG)

In addition to the front-end application, in all the notebooks there are charts to illustrate the analysis as well as the trainning of the models.

## Built with 🛠️

* Python
* Streamlit

⌨️ with ❤️ by [sarasabino](https://github.com/sarasabino) 😊
