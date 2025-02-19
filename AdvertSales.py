import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px

# st.title('Adverts and Sales')
# st.header('Built by Lola')
data = pd.read_csv('AdvertandSales.csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Bazooka'>ADVERT SALES PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by LOLA</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com.png')
st.divider()
st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown('Businesses struggle to predict sales and optimize ad spending due to changing market trends and customer behavior. Our Advert and Sales Prediction App uses AI and real-time data to forecast sales and improve ad performance, helping businesses maximize ROI and stay competitive.')

st.divider()

st.dataframe(data, use_container_width = True)

st.sidebar.image('pngwing(1).com.png', caption = 'Welcome User')

tv = st.sidebar.number_input('Television advert exp', min_value=0.0, max_value=10000.0, value=data.TV.median())
radio = st.sidebar.number_input('Radio advert exp', min_value=0.0, max_value=10000.0, value=data.Radio.median())
socials = st.sidebar.number_input('Social media exp', min_value= 0.0, max_value = 10000.0, value=data['Social Media'].median())
infl = st.sidebar.selectbox('Type of Influencer', data.Influencer.unique(),index=1)

inputs = {
    'TV' : [tv],
    'Radio' : [radio],
    'Social Media' : [socials],
    'Influencer' : [infl]
     }

inputVar = pd.DataFrame(inputs)
st.divider()
st.header('User Input')
st.dataframe(inputVar)

# transforming the user input, input the transformers
tv_scaler = joblib.load('TV_scaler.pkl')
radio_scaler = joblib.load('Radio_scaler.pkl')
social_scaler = joblib.load('Social Media_scaler.pkl')
influencer_encoder = joblib.load('Influencer_encoder.pkl')

# use the imported transformer to transform user input
inputVar['TV'] = tv_scaler.transform(inputVar[['TV']])
inputVar['Radio'] = radio_scaler.transform(inputVar[['Radio']])
inputVar['Social Media'] = social_scaler.transform(inputVar[['Social Media']])
inputVar['Influencer'] = influencer_encoder.transform(inputVar[['Influencer']])

# bringing in the model
model = joblib.load('Advertmodel.pkl')
predictbutton = st.button('Click to Predict the Sales')

if predictbutton:
    predicted = model.predict(inputVar)
    st.success(f'The predicted Sales value is: {predicted}')
