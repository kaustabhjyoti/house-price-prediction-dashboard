import streamlit as st 
import joblib
import pandas as pd  

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.sidebar.title("🏠 House Price Predictor")

st.sidebar.info(
    """
    This app predicts house prices using a Machine Learning model.
    Enter the house details and click **Predict House Price**.
    """
)

st.title("🏠 AI House Price Prediction")

st.markdown("### Predict house prices using Machine Learning")

st.divider()
st.markdown("## 🏡 Property Details")


left, right = st.columns(2)

with left:
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    grlivarea = st.number_input("Living Area (sq ft)",min_value=300, value=1500)

with right:
    garage_cars = st.number_input("Garage Cars", min_value=0, max_value=5, value=2)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000)





new_house = pd.DataFrame(0, index=[0], columns=model_columns)

new_house["OverallQual"] = overall_qual
new_house["GrLivArea"] = grlivarea
new_house["GarageCars"] = garage_cars
new_house["YearBuilt"] = year_built

st.divider()

if st.button(" 🚀 Predict House Price", use_container_width=True):
    st.markdown("## 📊 House Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="
            background: #1e1e1e;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 2px solid gold;">
            <h3>⭐</h3>
            <h4>Quality</h4>
            <h2>{overall_qual}/10</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: #1e1e1e;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 2px solid #00c853;">
            <h3>🏠</h3>
            <h4>Area</h4>
            <h2>{grlivarea:,} <span style="font-size: 18px;">ft²</span></h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: #1e1e1e;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 2px solid #bb86fc;">
            <h3>🚗</h3>
            <h4>Garage</h4>
            <h2>{garage_cars}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="
            background: #1e1e1e;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 2px solid #42a5f5;">
            <h3>🗓️</h3>
            <h4>Built</h4>
            <h2>{year_built}</h2>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    predicted_price = model.predict(new_house)
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1f4e79, #2e8b57);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            color: white;
            margin-top: 20px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.3);">
            <h3>💰 Estimated House Price</h3>
            <h1>${predicted_price[0]:,.0f}</h1>
            <p>AI Prediction Completed Successfully</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.balloons()

st.divider()
st.caption("Made with ❤️ by Kaustabhjyoti Baishya")



# st.markdown("## 📊 House Summary")
#     st.markdown(f"⭐ **Overall Quality:** {overall_qual}/10")
#     st.markdown(f"🏠 **Living Area:** {grlivarea:,} sq ft")
#     st.markdown(f"🚗 **Garage Cars:** {garage_cars}")
#     st.markdown(f"📅 **Year Built:** { year_built}")