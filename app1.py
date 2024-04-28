import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras import Sequential
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import requests
from bs4 import BeautifulSoup
import datetime



selected_option = st.sidebar.radio('Select an option:', ["Charts Suggestion's", "Market Sentiment", "Q & A"])

if selected_option == "Charts Suggestion's":
    feature_list = np.load('features_list.npy')

    filenames = np.load('filenames.npy')
    st.title("Nifty Chart Recommendation System")

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model.trainable = False

    model = Sequential([model, GlobalMaxPooling2D()])


    uploaded_file  = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224,224,3))
        img = img.resize((224, 224)) # Resize image
        image_array = np.asarray(img)  # Convert to NumPy array

        # Display Image (Optional)
        st.image(image_array, caption='Uploaded Image')
        expanded_img_array = np.expand_dims(image_array,axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        preds = model.predict(preprocessed_img)
        preds = preds.flatten()
        preds = preds/norm(preds)

        neighnours = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
        neighnours.fit(feature_list)

        distances,indices = neighnours.kneighbors([preds])

        for file in indices[0]:
            temp_img= image.load_img(filenames[file])
            st.image(temp_img)
        
elif selected_option == 'Market Sentiment':
        
    # pivot Values
    def daily_pivots():
        url = "https://www.topstockresearch.com/rt/Stock/NIFTY/PivotPoint"
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'} 
        webpage = requests.get(url,headers=headers).text
        soup = BeautifulSoup(webpage, 'lxml')
        table = soup.find_all('table')[0]
        row = table.find_all('tr')[1]
        data_cells = row.find_all('td')
        #type_value = data_cells[0].text.strip()
        s3 = data_cells[2].text.strip()
        s2 = data_cells[3].text.strip()
        s1 = data_cells[4].text.strip()
        pivot = data_cells[5].text.strip()
        r1 = data_cells[6].text.strip()
        r2 = data_cells[7].text.strip()
        r3 = data_cells[8].text.strip()
        table = soup.find_all('table')[1]
        row = table.find_all('tr')[1]
        cprdata = []
        for i in row.find_all('td'):
            i = i.text
            cprdata.append(i)

        cprdata = list(map(float, cprdata))
        cprdata= sorted(cprdata)
        bcp, cp, tcp = cprdata
        a = 0
        if bcp > tcp:
            a = tcp
            bcp = tcp
            tcp = a
        return float(bcp), float(cp), float(tcp),float(s1),float(s2),float(s3),float(r1),float(r2),float(r3)

    bcp, cp, tcp, s1,s2,s3,r1,r2,r3 = daily_pivots()

    # Weekly pivot Values
    def weekly_pivots():
        url = "https://www.topstockresearch.com/rt/Stock/NIFTY/PivotPoint"
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'} 
        webpage = requests.get(url,headers=headers).text
        soup = BeautifulSoup(webpage, 'lxml')
        table = soup.find_all('table')[2]
        row = table.find_all('tr')[1]
        data_cells = row.find_all('td')
        type_value = data_cells[0].text.strip()
        ws3 = data_cells[2].text.strip()
        ws2 = data_cells[3].text.strip()
        ws1 = data_cells[4].text.strip()
        wpivot = data_cells[5].text.strip()
        wr1 = data_cells[6].text.strip()
        wr2 = data_cells[7].text.strip()
        wr3 = data_cells[8].text.strip()
        table = soup.find_all('table')[3]
        row = table.find_all('tr')[1]
        cprdata = []
        for i in row.find_all('td'):
            i = i.text
            cprdata.append(i)

        cprdata = list(map(float, cprdata))
        cprdata= sorted(cprdata)
        wbcp, wcp, wtcp = cprdata
        if wbcp > wtcp:
            a = wtcp
            wbcp = wtcp
            wtcp = a
        return float(wbcp), float(wcp), float(wtcp), float(ws1), float(ws2), float(ws3), float(wr1), float(wr2), float(wr3)

    wbcp, wcp, wtcp, ws1, ws2, ws3, wr1, wr2, wr3 = weekly_pivots()

    def present_price_ema():
        url = "https://www.topstockresearch.com/rt/Stock/NIFTY/MovingAverage/Min5"
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'} 
        webpage = requests.get(url,headers=headers).text
        soup = BeautifulSoup(webpage, 'lxml')
        table = soup.find_all('table')[0]
        row = table.find_all('tr')[0]
        close = row.find_all('td')[1]
        table = soup.find_all('table')[1]
        row = table.find_all('tr')[4]
        ema = row.find_all('td')[0]
        return float(close.text), float(ema.text)

    # Weekly pivot Values
    def weekly_ema():
        url = "https://www.topstockresearch.com/rt/Stock/NIFTY/MovingAverage/Hour1"
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'} 
        webpage = requests.get(url,headers=headers).text
        soup = BeautifulSoup(webpage, 'lxml')
        table = soup.find_all('table')[1]
        row = table.find_all('tr')[4]
        ema = row.find_all('td')[0]
        return float(ema.text)


    present_price, present_ema = present_price_ema()
    hourly_ema = weekly_ema()

    st.header("Nifty Price")
    st.subheader(present_price)

    final_Score = 0

    if present_price > tcp:
        if present_price < r1:
            daily_pivot_score = 3
            st.write("Daily Pivots - TCP and R1 ", daily_pivot_score)
        elif present_price < r2:
            daily_pivot_score = 2
            st.write("Daily Pivots - R1 and R2 ", daily_pivot_score)
        elif present_price < r3:
            daily_pivot_score = 1
            st.write("Daily Pivots - R2 and R3 ", daily_pivot_score)
        else:
            daily_pivot_score = 0.5
            st.write("Daily Pivots - avove R3 ", daily_pivot_score)
    elif present_price < bcp:
        if present_price > r1:
            daily_pivot_score = -3
            st.write("Daily Pivots - BCP and S1 ", daily_pivot_score)
        elif present_price > r2:
            daily_pivot_score = -2
            st.write("Daily Pivots - S1 and S2 ", daily_pivot_score)
        elif present_price > r3:
            daily_pivot_score = -1
            st.write("Daily Pivots - S2 and S3 ", daily_pivot_score)
        else:
            daily_pivot_score = -0.5
            st.write("Daily Pivots - below S3 ", daily_pivot_score)
    else:
        daily_pivot_score = 0
        st.write("Daily Pivots - Central Pivot Range ","Warming Up")

    if present_price > wtcp:
        if present_price < wr1:
            weekly_pivot_score = 3 
            st.write("Weekly Pivots -  TCP and R1 ", weekly_pivot_score)
        elif present_price < wr2:
            weekly_pivot_score = 2
            st.write("Weekly Pivots - R1 and R2 ", weekly_pivot_score)
        elif present_price < wr3:
            weekly_pivot_score = 1
            st.write("Weekly Pivots - R2 and R3 ", weekly_pivot_score)
        else:
            weekly_pivot_score = 0.5
            st.write("Weekly Pivots - above R3 ", weekly_pivot_score)
    elif present_price < wbcp:
        if present_price > wr1:
            weekly_pivot_score = -3
            st.write("Weekly Pivots - BCP and S1 ", weekly_pivot_score)
        elif present_price > wr2:
            weekly_pivot_score = -2
            st.write("Weekly Pivots - S1 and S2 ", weekly_pivot_score)
        elif present_price > wr3:
            weekly_pivot_score = -1
            st.write("Weekly Pivots - S2 and S3 ", weekly_pivot_score)
        else:
            weekly_pivot_score = -0.5
            st.write("Weekly Pivots - S3 ", weekly_pivot_score)
    else:
        weekly_pivot_score = 0
        st.write("Weekly Pivots - Central Pivot Range ", "Warming Up")



    st.write('5 Min Chart EMA Difference -- ', round((present_price-present_ema),1) )

    st.write('1 Hour Chart EMA Difference -- ', round((present_price - hourly_ema),1)  )

    ema_diff_5min = present_price - present_ema
    ema_diff_hr = present_price - hourly_ema

    if ema_diff_5min >= 0 :
        if ema_diff_5min < 10:
            ema_score_5min = 3
        elif ema_diff_5min < 20 :
            ema_score_5min = 2
        else:
            ema_score_5min = 1
    else:
        if ema_diff_5min > -10:
            ema_score_5min = -3
        elif ema_diff_5min > -20 :
            ema_score_5min = -2
        else:
            ema_score_5min = -1

    if ema_diff_hr >= 0 :
        if ema_diff_hr < 50:
            ema_score_hr = 3
        elif ema_diff_hr < 100 :
            ema_score_hr = 2
        else:
            ema_score_hr = 1
    else:
        if ema_diff_hr > -50:
            ema_score_hr = -3
        elif ema_diff_hr > -100 :
            ema_score_hr = -2
        else:
            ema_score_hr = -1

    final_Score = round((daily_pivot_score + weekly_pivot_score + ema_score_5min+ema_score_hr)*100/12,2)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Success Probability  -->")
    with col2:
        st.subheader(final_Score)
    col1,col2,col3,col4 = st.columns(4)

    with st.form(" Details of the Day"):
        with col1:
            type_cpr = st.radio('Select the type of CPR  : ',('Wide', 'Medium', 'Narrow'))

        with col2:
            virgin_cpr = st.radio('Is there Virgin CPR for last 5 days : ',('Yes', 'No'))

        with col3:
            supply_demand_zones = st.radio('Is there any Supply or Demand Zones : ',('Yes', 'No'))

        with col4:
            any_gaps = st.radio('Is there any Gap regions : ',('Yes', 'No'))
        st.form_submit_button("Refresh")

    current_datetime = datetime.datetime.now() 
    hours = current_datetime.strftime("%H")
    minutes = current_datetime.strftime("%M")

    st.subheader("Possible Patterns")

    if int(hours) < 12 and int(minutes) < 15:
        col1, col2 = st.columns(2)
        with col1:
            st.write("1. OD (OPEN DRIVE) ")
            st.write("2. ODR(OD REJECTION) ")
            st.write("3. PPT(PIVOT PRESSURE TRADE) ")
            st.write("4. EVENING STAR ")
            st.write("5. MORNING STAR ")
        with col2:
            st.write("6. VIRGIN CPR RVRSL ")
            st.write("7. RCR (RED CANDLE RETRACEMENT) ")
            st.write("8. GCR(GREEN CANDLE RETRACEMENT) ")
            st.write("9. GAP UP REJECTION ")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("7. RCR (RED CANDLE RETRACEMENT) ")
            st.write("8. GCR(GREEN CANDLE RETRACEMENT) ")
            st.write("9. GAP UP REJECTION ")
            st.write("10. CPR BO(CENTRAL PIVOT RANGE BREAKOUT) ")
            st.write("11. GAP DOWN REJECTION  ")
        with col2:
            st.write("12. M RVRSL ")
            st.write("13. W RVRSL ")
            st.write("14. RCBO (RED CANDLE BREK OUT) ")
            st.write("15. GCBO(GREEN CANDLE BREAK OUT) ")

elif selected_option == 'Q & A':
    st.write('Under Development')


