import streamlit as st
import pandas as pd
import joblib   
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


st.set_page_config(page_title="Cancer Prediction",
                    page_icon="chart_with_upwards_trend")
filename = 'savmod.sav'

loaded_model = joblib.load(filename)

def hasil():
    data = ([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
            concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
            texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, 
            concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
            perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst])
    datas = pd.DataFrame([data], 
            columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst','compactness_worst', 
            'concavity_worst','concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])        
    
    bahan = pd.concat([datas,BC_clean])
    bahan = bahan.reset_index(drop=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(bahan)
    target = scaled[0,:]
    df_target = pd.DataFrame([target])
    if radius_mean == 0 :
        st.write("""Harap isi data dengan benar""")
        content.write("""
            # Harap isi data dengan benar""")
    else :

        result = loaded_model.predict_proba(df_target)
        labels =['Benign', 'Melignant']  
        fig1, ax1 = plt.subplots()
        ax1.pie(result[0,:], labels= labels, radius = 0.5, 
                autopct='%1.1f%%')
        with content:
            st.header('Result')
            st.pyplot(fig1)
content = st.container()
content.write("""
# Breast Cancer  Prediction

This app predicts the **Breast Cancer**!

Data obtained from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

""")


content.write("Sample data from Dataset")
BC_raw = pd.read_csv('Breast Cancer.csv')
BC = BC_raw.iloc[:,1:-1]
BC_clean = BC.iloc[:, 1:]
content.write(BC.sample(n=5,random_state=2))


st.sidebar.header('Input Data Manual')



with st.sidebar.expander('Click for Show/Hide') :
    form = st.form(key='my-form')
    radius_mean = form.number_input('radius_mean',step=.0001,format="%.4f")
    texture_mean = form.number_input('texture_mean',step=.0001,format="%.4f")
    perimeter_mean = form.number_input('perimeter_mean',step=.0001,format="%.4f")
    area_mean = form.number_input('area_mean',step=.0001,format="%.4f")
    smoothness_mean = form.number_input('smoothness_mean',step=.0001,format="%.4f")
    compactness_mean = form.number_input('compactness_mean',step=.0001,format="%.4f")
    concavity_mean = form.number_input('concavity_mean',step=.0001,format="%.4f")
    concave_points_mean = form.number_input('concave_points_mean',step=.0001,format="%.4f")
    symmetry_mean  = form.number_input('symmetry_mean',step=.0001,format="%.4f")
    fractal_dimension_mean = form.number_input('fractal_dimension_mean',step=.0001,format="%.4f")
    radius_se = form.number_input('radius_se',step=.0001,format="%.4f")
    texture_se = form.number_input('texture_se',step=.0001,format="%.4f")
    perimeter_se = form.number_input('perimeter_se',step=.0001,format="%.4f")
    area_se = form.number_input('area_se',step=.0001,format="%.4f")
    smoothness_se = form.number_input('smoothness_se',step=.0001,format="%.4f")
    compactness_se = form.number_input('compactness_se',step=.0001,format="%.4f")
    concavity_se = form.number_input('concavity_se',step=.0001,format="%.4f")
    concave_points_se = form.number_input('concave_points_se',step=.0001,format="%.4f")
    symmetry_se= form.number_input('symmetry_se',step=.0001,format="%.4f")
    fractal_dimension_se  = form.number_input('fractal_dimension_se',step=.0001,format="%.4f")
    radius_worst  = form.number_input('radius_worst',step=.0001,format="%.4f")
    texture_worst = form.number_input('texture_worst',step=.0001,format="%.4f")
    perimeter_worst = form.number_input('perimeter_worst',step=.0001,format="%.4f")
    area_worst = form.number_input('area_worst',step=.0001,format="%.4f")
    smoothness_worst = form.number_input('smoothness_worst',step=.0001,format="%.4f")
    compactness_worst = form.number_input('compactness_worst',step=.0001,format="%.4f")
    concavity_worst = form.number_input('concavity_worst',step=.0001,format="%.4f")
    concave_points_worst = form.number_input('concave_points_worst',step=.0001,format="%.4f")
    symmetry_worst = form.number_input('symmetry_worst',step=.0001,format="%.4f")
    fractal_dimension_worst = form.number_input('fractal_dimension_worst',step=.0001,format="%.4f")
    
    submit = form.form_submit_button('Submit')
    if submit :
        hasil()


uploaded_file = st.sidebar.file_uploader("Input using CSV File", type=["csv"])     


st.sidebar.write("""


This app Developed By 

**Tim 2**

""")





# Collects user input features into dataframe
