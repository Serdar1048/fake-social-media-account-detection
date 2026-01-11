import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Sahte Hesap Tespit Edici",
    page_icon="🕵️",
    layout="centered"
)

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_final_model():
    # Sadece Final Modeli (12 Özellikli) ve Scaler
    with open('models/final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_final_model()
except FileNotFoundError:
    st.error("Hata: Model dosyaları bulunamadı. Lütfen kurulumun doğru yapıldığından emin olun.")
    st.stop()

# --- BAŞLIK ---
st.title("🕵️ Sahte Hesap Tespit Edici")
st.markdown("### Profil Analizi")
st.caption("Analiz için profilin **en az 5 gönderiye** sahip olması gerekmektedir.")

st.markdown("---")

# --- GİRİŞ FORMU ---
with st.form("prediction_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Temel Bilgiler")
        
        pos = st.number_input("Toplam Gönderi Sayısı (pos)", min_value=0, value=10)
        
        if pos < 5:
            st.warning("⚠️ Sağlıklı bir analiz için hesapta en az 5 gönderi olmalıdır. 5 gönderi altındaki hesaplar analiz edilemez.")
            # Diğer inputları göstermeye veya işlemeye gerek yok
            submitted = st.form_submit_button("Analiz Yapılamaz", disabled=True)
        else:
            flw = st.number_input("Takipçi Sayısı (flw)", min_value=0, value=100)
            flg = st.number_input("Takip Edilen Sayısı (flg)", min_value=0, value=100)
            bl = st.number_input("Biyografi Karakter Sayısı (bl)", min_value=0, value=0)
            pic = st.selectbox("Profil Resmi Var mı? (pic)", [1, 0], format_func=lambda x: "Evet" if x == 1 else "Hayır")
            lin = st.selectbox("Biyografide Link Var mı? (lin)", [1, 0], format_func=lambda x: "Evet" if x == 1 else "Hayır")
            cl = st.number_input("Ortalama Açıklama Uzunluğu (cl)", min_value=0, value=10)

    with col2:
        if pos >= 5:
            st.subheader("📊 İçerik Detayları")
            
            video_count = st.number_input("Video/Reels Sayısı", min_value=0, value=0)
            loc_count = st.number_input("Konum Paylaşılan Gönderi Sayısı", min_value=0, value=0)
            hash_count = st.number_input("Hashtag Kullanılan Gönderi Sayısı", min_value=0, value=0)
            
            st.markdown("**Benzerlik Skoru (cs)**")
            cs = st.slider("Gönderi Benzerliği", 0.0, 1.0, 0.0, 0.01)
            
            st.markdown("**Paylaşım Sıklığı (pi)**")
            pi_val = st.number_input("Ortalama Paylaşım Aralığı", min_value=0.0, value=24.0)
            pi_unit = st.radio("Birim", ["Saat", "Gün"], horizontal=True)
            
    if pos >= 5:
        st.markdown("---")
        submitted = st.form_submit_button("🔍 Analiz Et", type="primary", use_container_width=True)

# --- TAHMİN MANTIĞI ---
if pos >= 5 and submitted:
    
    # 1. Hesaplamalar
    ni = video_count / pos
    lt = loc_count / pos
    hc = hash_count / pos 
    pi = pi_val * 24.0 if pi_unit == "Gün" else pi_val
    
    # 2. Veri Hazırlığı (12 Özellik)
    feature_cols = ['pos', 'flw', 'flg', 'bl', 'pic', 'lin', 'cl', 'ni', 'lt', 'hc', 'cs', 'pi']
    input_data = pd.DataFrame([{
        'pos': pos, 'flw': flw, 'flg': flg, 'bl': bl, 'pic': pic, 'lin': lin, 'cl': cl,
        'ni': ni, 'lt': lt, 'hc': hc, 'cs': cs, 'pi': pi
    }])
    
    # Sıralama ve Ölçeklendirme
    input_data = input_data[feature_cols]
    input_scaled = scaler.transform(input_data)
    
    # 3. Tahmin
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 4. Sonuç
    st.header("Sonuç")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if prediction == 1:
            st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=120)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=120)
            
    with col_res2:
        if prediction == 1:
            st.error(f"🚨 **SAHTE HESAP** tespit edildi.")
            st.metric("Risk Skoru", f"%{probability*100:.2f}")
        else:
            st.success(f"✅ **GERÇEK HESAP** olarak değerlendirildi.")
            st.metric("Güven Skoru", f"%{(1-probability)*100:.2f}")

    with st.expander("Hesaplanan Verileri Gör"):
        st.dataframe(input_data)
