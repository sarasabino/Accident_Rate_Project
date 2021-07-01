# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:04 2021

@author: sarasabino
"""
import streamlit as st
import pandas as pd

# Use the full page instead of a narrow central column
st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
 )

st.title("Accident Prediction Application")

DATA_URL = 'Data/staff_encoded.csv'         

## we are going to upload the data to show it on the interface

def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL,sep = ';')
    return data  

    
import altair as alt
data = load_data(DATA_URL)


data['Accident'] = data['Accident'].replace(1, 'Yes').replace(0, 'No')


st.subheader('Raw Data used in the trainings')
st.write(data)

st.subheader('Overview analysis from company staff')

staff = pd.read_csv('Data/Total_staff_by_employee.csv', sep=';')

staff['N employees'] = 1
staff['Severity'] = staff['Severity'].fillna('N/A')
staff['Tipo accidente'] = staff['Tipo accidente'].fillna('N/A')

data = staff.groupby(by=['Fabrica', 'Accident']).agg(sum)
data = data[['N employees']].reset_index()


chart = alt.Chart(data, title = 'Nº employees with/without accidents by Plant').mark_line().encode(x='Fabrica',y='N employees', color = 'Accident').properties(width=1000,height=300).interactive()



n_employees = staff.groupby(by=['Tecnología',
       'Accident', 'Severity']).agg('sum')

n_employees = n_employees.reset_index()
#n_employees.drop(columns={'Planta', 'Horas Presencia Efectiva Subcontratados','Horas Presencia Efectiva ETTs', 'Horas Formacion Seguridad Propios','Horas Formacion Seguridad ETTs'}, inplace=True)

n_employees_acc = n_employees.loc[n_employees['Accident']=='Yes']
n_employees_acc.rename(columns={'N employees':'N accidents', 'Tecnología':'Technology'}, inplace=True)

chart2 = alt.Chart(n_employees_acc,title="Nº Accidents by Technology and Sevirity").mark_line().encode( 
    y = 'N accidents',
    x ='Technology',
    color='Severity').interactive() .properties(
    width=1000,
    height=300
)

        
## displayin the charts in columns



col1, col2 = st.beta_columns(2)


with col1:
    
    st.altair_chart(chart,use_container_width=True)

with col2:
    st.altair_chart(chart2)
    
## ---------------- MAP
col3, col4 = st.beta_columns((2,1))



plantas = pd.read_csv('Data/G_Plantas y Tech_streamlit.csv', sep=';')

map_dt = data.merge(plantas, on=['Fabrica'])
map_dt.drop(columns={'ID', 'Activo', 'Tecnología'}, inplace=True)
map_dt = map_dt.groupby(['Fabrica', 'Accident', 'Pais', 'lat', 'lon']).agg(sum)
map_dt = map_dt.reset_index()

accidents = map_dt.loc[map_dt['Accident']=='Yes']
accidents.rename(columns={'N employees' : 'N accidents'}, inplace=True)


import folium
from folium import plugins
import streamlit_folium
from streamlit_folium import folium_static


m = folium.Map()

for (index, row) in accidents.iterrows():
    folium.Marker(location=[row.loc['lat'], row.loc['lon']],
                  popup = [row.loc['Pais'] + ' ' + 'Nº Accidents: ', row.loc['N accidents']]).add_to(m)

with col3:
    st.subheader('Nº accidents per country')
    folium_static(m)

## ---------- chart accidents
import matplotlib

import matplotlib.pyplot as plt
labels = 'Employees Accidents', 'Employees without accident'
sizes = [len(staff[staff['Accident']=='Yes']), len(staff[staff['Accident']=='No'])]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


with col4:
    st.pyplot(fig1)



### ----------- employees by country

n_employees = staff.groupby(by=['Fabrica', 'Gender','Tipo trabajador']).agg('sum')

n_employees = n_employees.reset_index()
n_employees.drop(columns={'Planta', 'Horas Presencia Efectiva Subcontratados',
       'Horas Presencia Efectiva ETTs', 'Horas Formacion Seguridad Propios',
       'Horas Formacion Seguridad ETTs'}, inplace=True)


chart3 = alt.Chart(n_employees, title="Nº employees by Fabric and Gender").mark_bar().encode( 
    y = 'N employees',
    x ='Fabrica',
     color='Gender').properties(
    width=1000,
    height=300
)
         
st.subheader('Employees by Fabric and Gender')

st.altair_chart(chart3,use_container_width=True)
