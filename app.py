import streamlit as st
import pandas as pd
import numpy as np
import joblib 

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Sahte Hesap Tespit Edici",
    page_icon="🕵️",
    layout="centered"
)

# --- CSS HACKS ---
st.markdown("""
<style>
/* Input içindeki 'Press Enter to apply' yazısını gizle */
div[data-testid="InputInstructions"] {
    display: none;
}
/* Textarea boyutlandırmasını kapat (Fixed height + Scroll) */
textarea, .stTextArea textarea {
    resize: none !important;
}
</style>
""", unsafe_allow_html=True)

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_final_model():
    # joblib.load sıkıştırılmış dosyaları otomatik algılar
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_final_model()
except FileNotFoundError:
    st.error("Hata: Model dosyaları bulunamadı.")
    st.stop()

# --- BAŞLIK ---
st.title("🕵️ Sahte Hesap Tespit Edici")
st.markdown("### Profil Analizi (Hibrit Analiz)")
st.caption("Verileri girdikten sonra en alttaki butona basarak analizi başlatın.")

# --- SESSION STATE (DİNAMİK YÖNETİM) ---
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# --- GİRİŞ ALANI ---

# 1. Post Sayısı (Manuel Giriş + Buton)
st.subheader("👤 Temel Bilgiler")

st.markdown("**Toplam Gönderi Sayısı (pos)**")
c_pos, c_btn_pos = st.columns([3, 1])
with c_pos:
    pos_str = st.text_input("pos_input", value="10", label_visibility="collapsed")
with c_btn_pos:
    if st.button("Uygula", key="btn_pos", use_container_width=True):
        pass

try:
    pos = int(pos_str)
except:
    pos = 10

# 5 Gönderi Kontrolü - Anlık Uyarı
if pos < 5:
    st.warning("⚠️ Analiz için en az 5 gönderi gereklidir!")

col1, col2 = st.columns(2)

with col1:
    
    # Takipçi (Input + Buton)
    st.markdown("**Takipçi Sayısı (flw)**")
    c_flw, c_btn_flw = st.columns([3, 1])
    with c_flw:
        flw_str = st.text_input("flw_input", value="100", label_visibility="collapsed")
    with c_btn_flw:
        st.button("Uygula", key="btn_flw", use_container_width=True)
        
    try: flw = int(flw_str)
    except: flw = 100

    # Takip Edilen (Input + Buton)
    st.markdown("**Takip Edilen Sayısı (flg)**")
    c_flg, c_btn_flg = st.columns([3, 1])
    with c_flg:
        flg_str = st.text_input("flg_input", value="100", label_visibility="collapsed")
    with c_btn_flg:
        st.button("Uygula", key="btn_flg", use_container_width=True)
        
    try: flg = int(flg_str)
    except: flg = 100
    
    # Biyografi (Text Area + Buton Altta)
    st.markdown("**Biyografi Metni**")
    bio_text = st.text_area("bio_input", height=100, help="Biyografiyi buraya yapıştırın.", label_visibility="collapsed")
    if st.button("Uygula", key="btn_bio", use_container_width=True):
        pass
        
    bl = len(bio_text)
    
    
    # --- CL GÜNCELLEMESİ (KATEGORİK) ---
    st.markdown("**Açıklama Tarzı (cl)**")
    cl_option = st.selectbox(
        "Gönderi Altı Açıklama Tarzı",
        [
            "Sadece Emoji / Çok Kısa (Örn: 🌊, ❤️)", 
            "Kısa Cümle (Örn: Harika bir gün.)", 
            "Orta (1-3 Cümle / Açıklayıcı)", 
            "Uzun (Hikaye / Detaylı Metin)"
        ]
    )
    
    if "Sadece Emoji" in cl_option: cl = 5 
    elif "Kısa Cümle" in cl_option: cl = 40 
    elif "Orta" in cl_option: cl = 150 
    else: cl = 400 
    
    # --- EKSTRA GÜVEN FAKTÖRLERİ (SOL SÜTUNA TAŞINDI) ---
    st.markdown("---")
    st.markdown("**🌟 Ekstra Güven Faktörleri**")
    
    # Dar alan olduğu için 2 sütunlu yapı
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        is_verified = st.checkbox("Mavi Tik", help="Onaylı hesap rozeti")
        has_highlights = st.checkbox("Öne Çıkanlar", help="Hikaye arşivi var mı?")
        has_carousel = st.checkbox("Kaydırmalı", help="Çoklu fotoğraf paylaşımı")
    with col_ex2:
        pic_check = st.checkbox("Profil Resmi", help="Profil fotoğrafı var mı?")
        lin_check = st.checkbox("Bio Linki", help="Biyografide link var mı?")

    pic = 1 if pic_check else 0
    lin = 1 if lin_check else 0


with col2:
    st.subheader("📊 İçerik Detayları")
    # Sliderların maksimum değeri Post sayısına (pos) eşitlenir.
    safe_max = pos if pos > 0 else 1
    
    video_count = st.slider("Video/Reels Sayısı", 0, safe_max, 0, key="vid_slider")
    loc_count = st.slider("Konum Paylaşılan", 0, safe_max, 0, key="loc_slider")
    hash_count = st.slider("Hashtag Kullanılan", 0, safe_max, 0, key="hash_slider")
    
    st.markdown("**Gönderi Benzerliği (cs)**")
    cs_percent = st.slider("Benzerlik Oranı (%)", 0, 100, 0, 1)
    cs = cs_percent / 100.0
    
    # Referans Tablosu
    with st.expander("ℹ️ Benzerlik Referans Tablosu"):
        st.markdown("""
        - **%0-20 (Benzersiz):** Birbirinden tamamen bağımsız içerikler.
        - **%20-40 (Düşük):** Aynı kişi/tema ama farklı ortamlar.
        - **%40-60 (Orta):** Aynı konsept ve renk tonları.
        - **%60-80 (Yüksek):** Seri çekim hissi veren kareler.
        - **%80-100 (Kopya/Bot):** Tıpatıp aynı görselin tekrarı.
        """)

st.markdown("---")

# --- ACTION BUTONU (Sadece ilk başlangıç için) ---
if not st.session_state.analysis_started:
    if st.button("🔍 Analizi Başlat", type="primary", use_container_width=True):
        st.session_state.analysis_started = True
        st.rerun()

# --- TAHMİN MANTIĞI (Dinamik) ---
if st.session_state.analysis_started:
    
    # 1. Validasyonlar
    if pos < 5:
        st.error("⚠️ Analiz için en az 5 gönderi gereklidir!")
        # Stop etmiyoruz, kullanıcı düzeltebilsin diye uyarı veriyoruz
    
    else:
        # İçerik sayıları Post sayısını geçemez (Görsel slider sınırlıyor ama her ihtimale karşı)
        video_count = min(video_count, pos)
        loc_count = min(loc_count, pos)
        hash_count = min(hash_count, pos)
        
        # 2. Veri Hazırlığı
        ni = video_count / pos
        lt = loc_count / pos
        hc = hash_count / pos 
        pi = 24.0 # Sabit
        
        feature_cols = ['pos', 'flw', 'flg', 'bl', 'pic', 'lin', 'cl', 'ni', 'lt', 'hc', 'cs', 'pi']
        input_data = pd.DataFrame([{
            'pos': pos, 'flw': flw, 'flg': flg, 'bl': bl, 'pic': pic, 'lin': lin, 'cl': cl,
            'ni': ni, 'lt': lt, 'hc': hc, 'cs': cs, 'pi': pi
        }])
        
        input_scaled = scaler.transform(input_data[feature_cols])
        
        # 3. Temel Model Tahmini
        base_probability = model.predict_proba(input_scaled)[0][1]
        
        # 4. Hibrit Puanlama
        final_risk_score = base_probability
        
        if is_verified: final_risk_score *= 0.85 
        if has_highlights: final_risk_score *= 0.85
        if has_carousel: final_risk_score *= 0.85
            
        # Final Karar
        prediction = 1 if final_risk_score > 0.50 else 0
        real_score_percent = (1 - final_risk_score) * 100
        
        # 5. Sonuç Gösterimi (Dinamik)
        st.header("Sonuç Analizi")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            if prediction == 1:
                st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
            else:
                st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)
                
        with col_res2:
            # Her durumda Gerçek Kişi Yüzdesini gösteriyoruz
            if prediction == 1:
                st.error(f"🚨 **RİSKLİ HESAP TESPİT EDİLDİ**")
                st.markdown(f"**Gerçek Kullanıcı Olma İhtimali:** %{real_score_percent:.1f}")
                st.caption(f"(Risk Skoru: %{final_risk_score*100:.1f})")
            else:
                st.success(f"✅ **GERÇEK HESAP**")
                st.markdown(f"**Gerçek Kullanıcı Olma İhtimali:** %{real_score_percent:.1f}")
                
        # 6. Akıllı Tavsiyeler
        if final_risk_score > 0.10:
            st.info("💡 **Güven Skorunu Artırmak İçin En Etkili 3 Adım:**")
            
            improvements = []
            
            if not is_verified: improvements.append({"msg": "Mavi Tik Almayı Dene (En Büyük Etki)", "score": 90})
            if not pic: improvements.append({"msg": "Profil Resmi Ekle (Çok Kritik)", "score": 85})
            if not has_highlights: improvements.append({"msg": "Hikayelerini Öne Çıkar (Aktiflik Göstergesi)", "score": 60})
            if not has_carousel: improvements.append({"msg": "Kaydırmalı Post Paylaş (Emek Göstergesi)", "score": 55})
            if not lin and flw > 1000: improvements.append({"msg": "Biyografine Link Ekle (Güven Verir)", "score": 40})
            
            if cs > 0.4:
                score = (cs - 0.2) * 100 
                improvements.append({"msg": "Gönderi Benzerliğini Azalt (Daha çeşitli fotoğraflar paylaş)", "score": score})
                
            if ni < 0.2: 
                score = (0.5 - ni) * 80 
                improvements.append({"msg": "Daha Fazla Video/Reels Paylaş", "score": score})
                
            if lt < 0.1:
                improvements.append({"msg": "Gönderilerine Konum Ekle", "score": 30})
                
            if cl < 20:
                 improvements.append({"msg": "Gönderi Açıklamalarını Uzat (Sadece emoji yetersiz)", "score": 25})
                 
            if flg > flw * 2:
                 improvements.append({"msg": "Takip Ettiklerini Azalt (Takipçi/Takip oranını dengele)", "score": 35})

            improvements.sort(key=lambda x: x['score'], reverse=True)
            
            for item in improvements[:3]:
                st.write(f"- 🚀 **{item['msg']}**")
