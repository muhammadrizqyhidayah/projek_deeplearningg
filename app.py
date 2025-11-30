import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import string
from io import StringIO
import csv
import requests
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ===== PREPROCESSING FUNCTIONS FROM NOTEBOOK =====
def remove_noise(text):
    """Remove URLs, mentions, hashtags, retweets, special characters"""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def to_lowercase(text):
    """Convert to lowercase"""
    return str(text).lower()

def tokenize_words(text):
    """Tokenize text into words"""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Remove Indonesian stopwords"""
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(['iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'])
    filtered = [token for token in tokens if token not in list_stopwords]
    return filtered

def stemmingText(text):
    """Apply stemming (not used in current pipeline but available)"""
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def reconstruct_text(list_words):
    """Convert list of words back to sentence"""
    return ' '.join(word for word in list_words)

# Slang dictionary from notebook
slangwords_notebook = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak",
    "udah": "sudah", "udh": "sudah", "dah": "sudah",
    "tp": "tapi", "tapi": "tetapi",
    "bgt": "banget", "banget": "sangat",
    "bgus": "bagus", "bgs": "bagus",
    "jelek": "buruk", "jlek": "buruk",
    "mantap": "bagus", "mantul": "bagus",
    "keren": "bagus", "ok": "oke", "oke": "baik",
    "thx": "terima kasih", "thanks": "terima kasih", "makasih": "terima kasih",
    "plz": "tolong", "pls": "tolong",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "sdh": "sudah",
    "hrs": "harus", "tdk": "tidak", "blm": "belum", "krn": "karena",
    "gmn": "bagaimana", "gimana": "bagaimana",
    "knp": "kenapa", "knapa": "kenapa",
    "emg": "memang", "emang": "memang",
    "cuma": "hanya", "cm": "hanya",
    "sih": "", "nih": "", "dong": "", "deh": "", "lah": "", "kah": "",
    "@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar",
    "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget",
    "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber",
    "cod": "bayar ditempat", "adlh": "adalah", "afaik": "as far as i know", "ahaha": "haha",
    "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku",
    "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur",
    "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial",
    "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik",
    "asekk": "asyik", "asem": "asam", "astul": "asal tulis", "ato": "atau",
    "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang",
    "b4": "sebelum", "bakalan": "akan", "bangedh": "banget", "begajulan": "nakal",
    "beliin": "belikan", "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga",
    "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang",
    "bgmn": "bagaimana", "bijimane": "bagaimana", "bkl": "akan", "bknnya": "bukannya",
    "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci",
    "bnran": "yang benar", "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar",
    "bokis": "bohong", "boljug": "boleh juga", "boyeh": "boleh", "br": "baru",
    "brg": "bareng", "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa",
    "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia",
    "bubbu": "tidur", "bubu": "tidur", "bw": "bawa", "bwt": "buat", "byk": "banyak",
    "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "can": "belum",
    "capcus": "pergi", "caper": "cari perhatian", "ce": "cewek", "cemen": "penakut",
    "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang",
    "cimeng": "ganja", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep",
    "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci",
    "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok",
    "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "cwek": "cewek",
    "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik",
    "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan", "diancurin": "dihancurkan",
    "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat",
    "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan",
    "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari",
    "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang",
    "duh": "aduh", "duren": "durian", "ed": "edisi", "egp": "emang gue pikirin",
    "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak",
    "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile",
    "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi",
    "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa",
    "gan": "juragan", "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja",
    "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas",
    "gile": "gila", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas",
    "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "gowes": "mengayuh sepeda",
    "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu",
    "gua": "saya", "guoblok": "goblok", "gw": "saya"
}

def normalize_slang_notebook(text):
    """Normalize slang words - notebook version"""
    words = text.split()
    fixed_words = [slangwords_notebook.get(word.lower(), word) for word in words]
    return ' '.join(fixed_words)
# ===== END PREPROCESSING FUNCTIONS =====

# Set page config
st.set_page_config(
    page_title="Analisis Sentimen Review APK",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Aplikasi Analisis Sentimen Review APK Play Store menggunakan Machine Learning"
    }
)

# Custom CSS for modern UI/UX - Responsive Design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Responsive Viewport */
    @viewport {
        width: device-width;
        zoom: 1.0;
    }
    
    /* Base responsive container */
    .main .block-container {
        max-width: 100%;
        padding: 1rem 2rem;
    }
    
    /* Main Container */
    .main {
        background: #ffffff;
        padding: 2rem;
    }
    
    .stApp {
        background: #f0f2f6;
    }
    
    /* Sidebar Styling - Modern Clean Design */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #1a252f 100%) !important;
        padding: 1.5rem 1rem;
    }
    
    /* All sidebar text - WHITE */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Radio button styling - Clean minimal cards */
    [data-testid="stSidebar"] .stRadio label {
        background: #ffffff !important;
        padding: 1rem 1.2rem !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    /* Force dark text on white cards */
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] .stRadio label div {
        color: #1a252f !important;
        font-weight: 700 !important;
    }
    
    /* Radio button hover */
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #f8f9fa !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    
    /* Selected radio - gradient background */
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5) !important;
        transform: translateY(-2px) scale(1.02) !important;
    }
    
    /* Force white text in selected state */
    [data-testid="stSidebar"] .stRadio label:has(input:checked) p,
    [data-testid="stSidebar"] .stRadio label:has(input:checked) span,
    [data-testid="stSidebar"] .stRadio label:has(input:checked) div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
        margin: 20px 0;
        border-left: 6px solid #667eea;
        transition: all 0.4s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.25);
        border-left-color: #764ba2;
    }
    
    /* Header Styling */
    h1 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        padding: 1rem 0;
    }
    
    h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #5a6c7d !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Text Input & Text Area Styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 12px !important;
        font-size: 1rem !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Success/Error/Warning Box Styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px !important;
        padding: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Dataframe Styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
    }
    
    /* Radio Button Styling */
    .stRadio>div {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Divider Styling */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #667eea, transparent) !important;
    }
    
    /* Section Container */
    .section-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Custom Header Box */
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .header-box h1 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-box p {
        color: #ffffff !important;
    }
    
    /* Footer Styling */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }
    
    /* Loading/Splash Screen Text */
    .appview-container .main .block-container {
        color: #ffffff !important;
    }
    
    /* Force white text on loading screen */
    div[data-testid="stAppViewContainer"] > div > div > div > div {
        color: #ffffff !important;
    }
    
    /* Streamlit default text on dark background */
    .stApp > header,
    .stApp [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* ============================================
       RESPONSIVE DESIGN - MOBILE & TABLET
       ============================================ */
    
    /* Mobile devices (portrait phones, less than 768px) */
    @media only screen and (max-width: 767px) {
        /* Container padding */
        .main .block-container {
            padding: 0.5rem 1rem !important;
        }
        
        /* Header box responsive */
        .header-box {
            padding: 1.5rem 1rem !important;
            margin-bottom: 1rem !important;
        }
        
        .header-box h1 {
            font-size: 1.8rem !important;
        }
        
        .header-box p {
            font-size: 0.95rem !important;
        }
        
        /* Metric cards */
        .metric-card {
            padding: 1rem !important;
            margin: 0.8rem 0 !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
        
        /* Section container */
        .section-container {
            padding: 1rem !important;
            margin: 1rem 0 !important;
        }
        
        /* Buttons */
        .stButton>button {
            padding: 0.6rem 1.5rem !important;
            font-size: 0.9rem !important;
            width: 100% !important;
        }
        
        /* Text input */
        .stTextArea textarea {
            font-size: 0.9rem !important;
        }
        
        /* Sidebar on mobile */
        [data-testid="stSidebar"] {
            padding: 1rem 0.5rem !important;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            padding: 0.8rem 1rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Hide sidebar by default on mobile */
        [data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: -21rem;
        }
        
        /* Columns stack on mobile */
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        
        /* Footer */
        .footer {
            padding: 1.5rem 1rem !important;
            font-size: 0.85rem !important;
        }
    }
    
    /* Tablet devices (768px to 1024px) */
    @media only screen and (min-width: 768px) and (max-width: 1024px) {
        .main .block-container {
            padding: 1rem 1.5rem !important;
        }
        
        .header-box {
            padding: 2rem 1.5rem !important;
        }
        
        .header-box h1 {
            font-size: 2rem !important;
        }
        
        .metric-card {
            padding: 1.5rem !important;
        }
        
        .section-container {
            padding: 1.5rem !important;
        }
        
        [data-testid="stSidebar"] {
            padding: 1.5rem 0.8rem !important;
        }
    }
    
    /* Large desktop (1920px and up) */
    @media only screen and (min-width: 1920px) {
        .main .block-container {
            max-width: 1600px !important;
            margin: 0 auto !important;
        }
        
        .header-box h1 {
            font-size: 3rem !important;
        }
        
        .header-box p {
            font-size: 1.4rem !important;
        }
    }
    
    /* Landscape orientation adjustments */
    @media only screen and (max-height: 600px) and (orientation: landscape) {
        .header-box {
            padding: 1rem !important;
            margin-bottom: 1rem !important;
        }
        
        .section-container {
            padding: 1rem !important;
            margin: 0.8rem 0 !important;
        }
    }
    
    /* Touch device optimizations */
    @media (hover: none) and (pointer: coarse) {
        /* Larger touch targets */
        .stButton>button {
            min-height: 48px !important;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            min-height: 48px !important;
        }
        
        /* Better spacing for touch */
        .stRadio label {
            margin: 0.6rem 0 !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

from pathlib import Path

# ========================================
# FUNGSI PREPROCESSING LENGKAP
# ========================================

def remove_noise(text):
    """Hapus noise seperti URL, mention, hashtag, emoticon"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)
    # Hapus hashtag (#)
    text = re.sub(r'#\w+', '', text)
    # Hapus email
    text = re.sub(r'\S+@\S+', '', text)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus karakter khusus dan punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def to_lowercase(text):
    """Convert text ke lowercase"""
    return str(text).lower()

def tokenize_words_custom(text):
    """Tokenisasi kata"""
    return word_tokenize(text)

def remove_stopwords_custom(tokens):
    """Hapus stopwords Indonesia dan Inggris"""
    # Stopwords Indonesia
    indonesian_stopwords = set([
        'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
        'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali', 'dan', 'tidak', 'ini', 'karena',
        'oleh', 'itu', 'dalam', 'dari', 'dengan', 'di', 'ada', 'akan', 'sudah', 'bisa', 'dapat',
        'saat', 'hanya', 'atau', 'juga', 'setelah', 'mereka', 'saya', 'kamu', 'kami', 'kita'
    ])
    
    # Stopwords English
    english_stopwords = set(stopwords.words('english'))
    
    # Gabungkan
    all_stopwords = indonesian_stopwords.union(english_stopwords)
    
    # Filter tokens
    filtered = [token for token in tokens if token.lower() not in all_stopwords and len(token) > 2]
    return filtered

def normalize_slang(text):
    """Normalisasi slang words Indonesia"""
    slang_dict = {
        'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
        'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
        'tp': 'tapi', 'tapi': 'tetapi',
        'bgt': 'banget', 'banget': 'sangat',
        'bgus': 'bagus', 'bgs': 'bagus',
        'jelek': 'buruk', 'jlek': 'buruk',
        'mantap': 'bagus', 'mantul': 'bagus',
        'keren': 'bagus', 'ok': 'oke', 'oke': 'baik',
        'thx': 'terima kasih', 'thanks': 'terima kasih', 'makasih': 'terima kasih',
        'plz': 'tolong', 'pls': 'tolong',
        'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'sdh': 'sudah',
        'hrs': 'harus', 'tdk': 'tidak', 'blm': 'belum', 'krn': 'karena',
        'gmn': 'bagaimana', 'gimana': 'bagaimana',
        'knp': 'kenapa', 'knapa': 'kenapa',
        'emg': 'memang', 'emang': 'memang',
        'cuma': 'hanya', 'cm': 'hanya',
        'sih': '', 'nih': '', 'dong': '', 'deh': '', 'lah': '', 'kah': '',
    }
    
    words = text.split()
    normalized_words = [slang_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)

def reconstruct_text(tokens):
    """Gabungkan tokens kembali jadi text"""
    return ' '.join(tokens)

def full_preprocess_pipeline(text):
    """Pipeline preprocessing lengkap"""
    if pd.isna(text) or text == '':
        return ""
    text = remove_noise(text)
    text = to_lowercase(text)
    text = normalize_slang(text)
    tokens = tokenize_words_custom(text)
    tokens = remove_stopwords_custom(tokens)
    text = reconstruct_text(tokens)
    return text

# Lexicon Scoring - SAMA SEPERTI MODEL.PY
@st.cache_data
def load_lexicon():
    """Load lexicon dari GitHub (sama seperti Model.py)"""
    import csv
    import requests
    from io import StringIO
    
    # Load positive lexicon
    lexicon_positive = dict()
    try:
        response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
        if response.status_code == 200:
            reader = csv.reader(StringIO(response.text), delimiter=',')
            for row in reader:
                if len(row) >= 2:
                    lexicon_positive[row[0]] = int(row[1])
    except Exception as e:
        st.warning(f"⚠️ Gagal load positive lexicon: {e}")
    
    # Load negative lexicon
    lexicon_negative = dict()
    try:
        response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
        if response.status_code == 200:
            reader = csv.reader(StringIO(response.text), delimiter=',')
            for row in reader:
                if len(row) >= 2:
                    lexicon_negative[row[0]] = int(row[1])
    except Exception as e:
        st.warning(f"⚠️ Gagal load negative lexicon: {e}")
    
    return lexicon_positive, lexicon_negative

def sentiment_analysis_lexicon_indonesia(tokens, lexicon_positive, lexicon_negative):
    """Hitung sentiment score berdasarkan lexicon Indonesia (sama seperti Model.py)"""
    score = 0
    
    # Hitung score dari positive words
    for word in tokens:
        if word in lexicon_positive:
            score = score + lexicon_positive[word]
    
    # Hitung score dari negative words  
    for word in tokens:
        if word in lexicon_negative:
            score = score + lexicon_negative[word]
    
    # Tentukan polarity
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    
    return score, polarity

# Load data dan models
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('data/ulasan_aplikasi_labelled.csv')
        return df
    except FileNotFoundError:
        st.error("Data tidak ditemukan! Pastikan file 'data/ulasan_aplikasi_labelled.csv' ada.")
        return None

@st.cache_resource
def load_models():
    base = Path(__file__).parent
    model_dir = base / "model"

    files = {
        "svm": model_dir / "svm_model.pkl",
        "tfidf": model_dir / "tfidf_vectorizer.pkl",
        "pipeline": model_dir / "pipeline_best.pkl",
        "le": model_dir / "label_encoder.pkl",
    }

    # cek eksistensi dan ukuran file
    for name, path in files.items():
        if not path.exists():
            st.error(f"Model file not found: {path}")
            return None, None, None, None
        if path.stat().st_size == 0:
            st.error(f"Model file appears empty/corrupt: {path}")
            return None, None, None, None

    def safe_load(path):
        # coba joblib.load dulu
        try:
            return joblib.load(path)
        except Exception as e_joblib:
            # fallback ke pickle sebagai opsi terakhir
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e_pickle:
                # tampilkan kedua error untuk debugging
                st.error(f"Failed to load {path} with joblib ({e_joblib}) and pickle ({e_pickle})")
                return None

    svm_model = safe_load(files["svm"])
    tfidf_vectorizer = safe_load(files["tfidf"])
    pipeline_model = safe_load(files["pipeline"])
    label_encoder = safe_load(files["le"])

    # jika salah satu gagal, kembalikan None agar app menampilkan error
    if any(x is None for x in [svm_model, tfidf_vectorizer, pipeline_model, label_encoder]):
        st.error("Gagal memuat semua model. Periksa log di atas.")
        return None, None, None, None

    return svm_model, tfidf_vectorizer, pipeline_model, label_encoder

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, float):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

# Load data and models
df = load_data()
svm_model, tfidf_vectorizer, pipeline_model, label_encoder = load_models()

# Initialize session state for accuracy variables
if 'svm_accuracy' not in st.session_state:
    st.session_state.svm_accuracy = None
if 'pipeline_accuracy' not in st.session_state:
    st.session_state.pipeline_accuracy = None

# Custom loading screen styling
st.markdown("""
    <style>
    /* Override Streamlit loading text color */
    .stApp::before {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("""
    <div style='text-align: center; padding: 2rem 1rem; margin-bottom: 2rem;'>
        <h1 style='color: #ffffff; font-size: 3rem; margin: 0;'>📱</h1>
        <h2 style='color: #ffffff; font-size: 1.4rem; margin-top: 1rem; font-weight: 700;'>
            Analisis Sentimen
        </h2>
        <p style='color: #ffffff; font-size: 0.95rem; margin-top: 0.5rem; font-weight: 500; opacity: 0.9;'>
            Review APK Play Store
        </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border: none; height: 1px; background: rgba(255,255,255,0.2); margin: 1.5rem 0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #ffffff; margin-bottom: 1.5rem; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;'>Menu</h3>", unsafe_allow_html=True)
page = st.sidebar.radio("", 
    ["📊 Dashboard", "🔍 Evaluasi Model", "🎲 Prediksi Sentimen", "📈 Data Overview", "📂 Upload CSV"],
    label_visibility="collapsed"
)

# ====================
# PAGE 1: DASHBOARD
# ====================
if page == "📊 Dashboard":
    st.markdown("""
        <div class='header-box'>
            <h1 style='margin: 0; font-size: 2.5rem;'>📱 Sentiment Analysis</h1>
            <p style='margin-top: 0.5rem; font-size: 1.2rem; opacity: 0.9;'>APK Play Store Reviews Analyzer</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Section
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='color: #667eea; margin: 0; font-size: 0.9rem;'>📚 TOTAL REVIEWS</h3>
                    <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{}</p>
                    <p style='color: #7f8c8d; font-size: 0.85rem; margin: 0;'>Reviews dalam dataset</p>
                </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            if 'polarity' in df.columns:
                st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #667eea; margin: 0; font-size: 0.9rem;'>🎯 UNIQUE SENTIMENTS</h3>
                        <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{}</p>
                        <p style='color: #7f8c8d; font-size: 0.85rem; margin: 0;'>Kategori sentimen</p>
                    </div>
                """.format(df['polarity'].nunique()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='color: #667eea; margin: 0; font-size: 0.9rem;'>⚠️ MISSING DATA</h3>
                    <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{}</p>
                    <p style='color: #7f8c8d; font-size: 0.85rem; margin: 0;'>Reviews kosong</p>
                </div>
            """.format(df['content'].isna().sum()), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Information Section
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📝 Informasi Proyek</h2>", unsafe_allow_html=True)
    
    st.write("""
    Aplikasi ini menganalisis sentimen dari review APK di Play Store menggunakan dua model machine learning yang canggih dan teruji.
    """)
    
    st.markdown("<h3 style='color: #667eea; margin-top: 2rem;'>🤖 Model yang Digunakan:</h3>", unsafe_allow_html=True)
    
    # Model 1
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; 
                border-left: 4px solid #667eea;'>
        <h4 style='color: #2c3e50; margin-top: 0;'>1. SVM (Support Vector Machine) + TF-IDF Vectorizer</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("- Optimal untuk klasifikasi biner dan multi-kelas")
    st.markdown("- Performa stabil dengan dataset terbatas")
    st.markdown("- Menggunakan kernel untuk non-linear classification")
    
    # Model 2
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                padding: 1.5rem; border-radius: 10px; margin-top: 1rem;
                border-left: 4px solid #ff9966;'>
        <h4 style='color: #2c3e50; margin-top: 0;'>2. Logistic Regression dengan Advanced Pipeline</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("- Pipeline: TF-IDF → SelectKBest → TruncatedSVD → LogisticRegression")
    st.markdown("- Menggunakan feature selection untuk efisiensi")
    st.markdown("- Dimensionality reduction untuk performa optimal")
    
    st.markdown("<h3 style='color: #667eea; margin-top: 2rem;'>✨ Fitur Aplikasi:</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 3px solid #667eea;'>
            <h4 style='color: #2c3e50; margin: 0;'>📊 Evaluasi Model</h4>
            <p style='color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;'>
                Analisis performa dengan metrics lengkap
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 3px solid #764ba2;'>
            <h4 style='color: #2c3e50; margin: 0;'>🎲 Prediksi Sentimen</h4>
            <p style='color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;'>
                Prediksi real-time dengan probability scores
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 3px solid #667eea;'>
            <h4 style='color: #2c3e50; margin: 0;'>📈 Data Overview</h4>
            <p style='color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;'>
                Visualisasi distribusi data yang interaktif
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ====================
# PAGE 2: MODEL EVALUATION
# ====================
elif page == "🔍 Evaluasi Model":
    st.markdown("""
        <div class='header-box'>
            <h1 style='margin: 0; font-size: 2.5rem;'>🔍 Evaluasi Model</h1>
            <p style='margin-top: 0.5rem; font-size: 1.2rem; opacity: 0.9;'>Analisis Performa Model Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None and svm_model is not None and pipeline_model is not None:
        # ======================
        # 1. Siapkan data uji
        # ======================
        X = df['final_text'].fillna('').apply(preprocess_text)
        y = df['polarity']
        
        # Encode labels ke numerik
        y_encoded = label_encoder.transform(y)
        
        # Split data (konsisten di semua evaluasi)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        
        # ======================
        # 2. Evaluasi Kedua Model (Kiri-Kanan)
        # ======================
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        
        # Transform test data dengan TF-IDF untuk SVM
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        svm_pred = svm_model.predict(X_test_tfidf)
        
        # Metrik SVM
        svm_accuracy = accuracy_score(y_test, svm_pred)
        st.session_state.svm_accuracy = svm_accuracy
        svm_precision = precision_score(y_test, svm_pred, average='weighted', zero_division=0)
        svm_recall = recall_score(y_test, svm_pred, average='weighted', zero_division=0)
        svm_f1 = f1_score(y_test, svm_pred, average='weighted', zero_division=0)
        
        # Pipeline prediksi untuk Logistic Regression
        pipeline_pred = pipeline_model.predict(X_test)
        pipeline_pred_encoded = label_encoder.transform(pipeline_pred) if isinstance(pipeline_pred[0], str) else pipeline_pred
        
        # Metrik Logistic Regression
        pipeline_accuracy = accuracy_score(y_test, pipeline_pred_encoded)
        st.session_state.pipeline_accuracy = pipeline_accuracy
        pipeline_precision = precision_score(y_test, pipeline_pred_encoded, average='weighted', zero_division=0)
        pipeline_recall = recall_score(y_test, pipeline_pred_encoded, average='weighted', zero_division=0)
        pipeline_f1 = f1_score(y_test, pipeline_pred_encoded, average='weighted', zero_division=0)
        
        # Layout Kiri-Kanan
        left_col, right_col = st.columns(2)
        
        # ===== KOLOM KIRI: SVM =====
        with left_col:
            st.markdown("<h2 style='color: #667eea; text-align: center;'>🤖 SVM + TF-IDF</h2>", unsafe_allow_html=True)
            
            # Metrik SVM
            st.markdown("<h3 style='color: #2c3e50; margin-top: 1rem;'>📈 Metrik</h3>", unsafe_allow_html=True)
            st.metric("Accuracy", f"{svm_accuracy:.4f}")
            st.metric("Precision", f"{svm_precision:.4f}")
            st.metric("Recall", f"{svm_recall:.4f}")
            st.metric("F1-Score", f"{svm_f1:.4f}")
            
            # Confusion Matrix SVM
            st.markdown("<h3 style='color: #2c3e50; margin-top: 1.5rem;'>📊 Confusion Matrix</h3>", unsafe_allow_html=True)
            svm_cm = confusion_matrix(y_test, svm_pred, labels=range(len(label_encoder.classes_)))
            fig_svm, ax_svm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                svm_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax_svm
            )
            ax_svm.set_ylabel('True Label')
            ax_svm.set_xlabel('Predicted Label')
            st.pyplot(fig_svm)
            plt.close()
        
        # ===== KOLOM KANAN: LOGISTIC REGRESSION =====
        with right_col:
            st.markdown("<h2 style='color: #ff9966; text-align: center;'>🤖 Logistic Regression</h2>", unsafe_allow_html=True)
            
            # Metrik Logistic Regression
            st.markdown("<h3 style='color: #2c3e50; margin-top: 1rem;'>📈 Metrik</h3>", unsafe_allow_html=True)
            st.metric("Accuracy", f"{pipeline_accuracy:.4f}")
            st.metric("Precision", f"{pipeline_precision:.4f}")
            st.metric("Recall", f"{pipeline_recall:.4f}")
            st.metric("F1-Score", f"{pipeline_f1:.4f}")
            
            # Confusion Matrix Logistic Regression
            st.markdown("<h3 style='color: #2c3e50; margin-top: 1.5rem;'>📊 Confusion Matrix</h3>", unsafe_allow_html=True)
            pipeline_cm = confusion_matrix(y_test, pipeline_pred_encoded, labels=range(len(label_encoder.classes_)))
            fig_pipeline, ax_pipeline = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                pipeline_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax_pipeline
            )
            ax_pipeline.set_ylabel('True Label')
            ax_pipeline.set_xlabel('Predicted Label')
            st.pyplot(fig_pipeline)
            plt.close()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ======================
        # 4. Tabel Perbandingan Metrik
        # ======================
        st.markdown("""
            <div class='section-container'>
                <h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📊 Perbandingan Model</h2>
        """, unsafe_allow_html=True)
        comparison_data = {
            'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'SVM': [svm_accuracy, svm_precision, svm_recall, svm_f1],
            'Logistic Regression': [pipeline_accuracy, pipeline_precision, pipeline_recall, pipeline_f1]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        x = np.arange(len(comparison_data['Metrik']))
        width = 0.35
        ax_comp.bar(x - width/2, comparison_data['SVM'], width, label='SVM', alpha=0.8)
        ax_comp.bar(x + width/2, comparison_data['Logistic Regression'], width, label='Logistic Regression', alpha=0.8)
        ax_comp.set_xlabel('Metrik')
        ax_comp.set_ylabel('Score')
        ax_comp.set_title('Perbandingan Performa Model')
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(comparison_data['Metrik'])
        ax_comp.legend()
        ax_comp.set_ylim([0, 1.1])
        plt.tight_layout()
        st.pyplot(fig_comp)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # ======================
        # 5. ROC Curve SVM vs Logistic Regression
        # ======================
        st.markdown("""
            <div class='section-container'>
                <h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📉 ROC Curve: SVM vs Logistic Regression</h2>
        """, unsafe_allow_html=True)

        try:
            # ===========================
            # SVM: pakai kelas svm_model
            # ===========================
            classes_svm = svm_model.classes_  # contoh: array([0, 1, 2])
            y_test_bin_svm = label_binarize(y_test, classes=classes_svm)

            # score SVM (decision_function) di data uji yang sudah di-TFIDF
            svm_scores = svm_model.decision_function(X_test_tfidf)
            # kalau binary, bentuknya (n_samples,), kita jadikan (n_samples, 1)
            if svm_scores.ndim == 1:
                svm_scores = svm_scores.reshape(-1, 1)

            # ROC micro-average (flatten semua kelas)
            fpr_svm, tpr_svm, _ = roc_curve(y_test_bin_svm.ravel(),
                                            svm_scores.ravel())
            auc_svm = auc(fpr_svm, tpr_svm)

            # ==========================================
            # Logistic Regression: pakai kelas pipeline
            # ==========================================
            lr_proba = pipeline_model.predict_proba(X_test)
            classes_lr = pipeline_model.classes_  # bisa string / angka

            # Samakan tipe y_test dengan classes_lr
            if isinstance(classes_lr[0], str):
                # y_test awalnya angka (hasil label_encoder) -> balik ke label string
                y_test_lr = label_encoder.inverse_transform(y_test)
            else:
                # sudah angka, langsung pakai
                y_test_lr = y_test

            y_test_bin_lr = label_binarize(y_test_lr, classes=classes_lr)

            fpr_lr, tpr_lr, _ = roc_curve(y_test_bin_lr.ravel(),
                                          lr_proba.ravel())
            auc_lr = auc(fpr_lr, tpr_lr)

            # ===========================
            # Plot ROC comparison
            # ===========================
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.3f})")
            ax_roc.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
            ax_roc.plot([0, 1], [0, 1], 'k--', label="Random")

            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve Comparison: SVM vs Logistic Regression")
            ax_roc.legend(loc="lower right")

            plt.tight_layout()
            st.pyplot(fig_roc)

        except Exception as e:
            st.warning(f"⚠️ Gagal menghitung ROC Curve: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
        # ======================
        # 6. Model terbaik
        # ======================
        best_model = "SVM" if svm_accuracy > pipeline_accuracy else "Logistic Regression"
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        box-shadow: 0 10px 30px rgba(86, 171, 47, 0.3); margin-top: 2rem;'>
                <h2 style='color: white; margin: 0;'>✅ Model Terbaik</h2>
                <h3 style='color: white; margin: 0.5rem 0 0 0; font-size: 2rem;'>{best_model}</h3>
                <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>
                    Accuracy: {max(svm_accuracy, pipeline_accuracy):.4f}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Data atau model tidak dapat dimuat!")


# ====================
# PAGE 3: PREDICTION
# ====================
elif page == "🎲 Prediksi Sentimen":
    st.markdown("""
        <div class='header-box'>
            <h1 style='margin: 0; font-size: 2.5rem;'>🔮 Prediksi Sentimen Review</h1>
            <p style='margin-top: 0.5rem; font-size: 1.2rem; opacity: 0.9;'>Analisis Sentimen Real-Time dengan AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    if svm_model is not None and pipeline_model is not None:
        # Input Section dengan styling modern
        st.markdown("""
            <div class='section-container' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-top: 1.5rem;'>
                <h2 style='color: white; margin: 0 0 1rem 0; font-size: 1.5rem;'>💬 Masukkan Review Anda</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Text input dengan styling
        review_input = st.text_area(
            "📝 Tulis review aplikasi di sini:", 
            placeholder="Contoh: Aplikasi ini sangat bagus dan mudah digunakan! Fitur-fiturnya lengkap.",
            height=150,
            help="Masukkan teks review dalam bahasa Indonesia untuk analisis sentimen"
        )
        
        # Model selection dengan UI yang lebih baik
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h3 style='color: #667eea; margin: 0 0 1rem 0; font-size: 1.2rem;'>🎯 Pilih Model untuk Prediksi:</h3>
            </div>
        """, unsafe_allow_html=True)
        
        model_choice = st.radio(
            "",
            ["SVM + TF-IDF", "Logistic Regression (Pipeline)"],
            help="SVM umumnya lebih akurat untuk dataset kecil, Logistic Regression lebih cepat untuk dataset besar"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Button dengan styling modern
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_clicked = st.button(
                "🔮 Analisis Sentimen Sekarang", 
                key="predict_button",
                use_container_width=True,
                type="primary"
            )
        
        if predict_clicked:
            if review_input.strip() == "":
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #f5af19 0%, #f12711 100%); 
                                padding: 1rem; border-radius: 10px; text-align: center; margin-top: 1rem;'>
                        <h3 style='color: white; margin: 0;'>⚠️ Silakan masukkan review terlebih dahulu!</h3>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Loading animation
                with st.spinner('🔄 Sedang menganalisis sentimen...'):
                    import time
                    time.sleep(0.5)
                    
                    # Preprocess input
                    processed_review = preprocess_text(review_input)
                
                if model_choice == "SVM + TF-IDF":
                    # SVM Prediction
                    X_input_tfidf = tfidf_vectorizer.transform([processed_review])
                    prediction_encoded = svm_model.predict(X_input_tfidf)[0]
                    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                    
                    # Get decision scores
                    decision_scores = svm_model.decision_function(X_input_tfidf)[0]
                    
                    # Softmax to get probabilities
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    probabilities = exp_scores / exp_scores.sum()
                    
                    # Determine sentiment color and emoji
                    sentiment_config = {
                        'positive': {'color': '#2ecc71', 'gradient': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)', 'emoji': '😊', 'icon': '✅'},
                        'negative': {'color': '#e74c3c', 'gradient': 'linear-gradient(135deg, #eb3349 0%, #f45c43 100%)', 'emoji': '😞', 'icon': '❌'},
                        'neutral': {'color': '#3498db', 'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'emoji': '😐', 'icon': '➖'}
                    }
                    
                    config = sentiment_config.get(prediction.lower(), sentiment_config['neutral'])
                    max_prob = probabilities.max()
                    
                    # Display result in a modern card
                    st.markdown(f"""
                        <div style='background: {config['gradient']}; 
                                    padding: 2rem; border-radius: 15px; text-align: center; 
                                    box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 2rem 0;'>
                            <h2 style='color: white; margin: 0; font-size: 1.2rem; opacity: 0.9;'>🤖 Model: SVM + TF-IDF</h2>
                            <h1 style='color: white; margin: 1rem 0; font-size: 3rem;'>{config['emoji']}</h1>
                            <h2 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>{config['icon']} {prediction.upper()}</h2>
                            <p style='color: white; margin: 1rem 0 0 0; font-size: 1.3rem; opacity: 0.95;'>Confidence: {max_prob*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probabilities for each class
                    prob_data = {
                        'Sentimen': label_encoder.classes_,
                        'Probability': probabilities
                    }
                    prob_df = pd.DataFrame(prob_data)
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Modern probability display
                    st.markdown("""
                        <div class='section-container'>
                            <h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📊 Distribusi Probabilitas</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create columns for each sentiment
                    cols = st.columns(len(prob_df))
                    for idx, (col, row) in enumerate(zip(cols, prob_df.itertuples())):
                        sentiment_name = row.Sentimen
                        sentiment_prob = row.Probability
                        sent_config = sentiment_config.get(sentiment_name.lower(), sentiment_config['neutral'])
                        
                        with col:
                            st.markdown(f"""
                                <div style='background: white; padding: 1.5rem; border-radius: 10px; 
                                            box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center;
                                            border-top: 4px solid {sent_config['color']};'>
                                    <h3 style='color: {sent_config['color']}; margin: 0; font-size: 1rem;'>{sent_config['emoji']} {sentiment_name.upper()}</h3>
                                    <p style='font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{sentiment_prob*100:.1f}%</p>
                                    <div style='background: #ecf0f1; border-radius: 10px; height: 8px; margin-top: 0.5rem;'>
                                        <div style='background: {sent_config['color']}; width: {sentiment_prob*100}%; height: 100%; border-radius: 10px;'></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Add bar chart visualization for SVM
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                    
                    # Prepare colors for each sentiment
                    bar_colors = [sentiment_config.get(sent.lower(), sentiment_config['neutral'])['color'] 
                                  for sent in prob_df['Sentimen']]
                    
                    # Create horizontal bar chart
                    bars = ax_bar.barh(prob_df['Sentimen'], prob_df['Probability'], color=bar_colors, height=0.6)
                    
                    # Styling
                    ax_bar.set_xlabel('Probability', fontsize=12, fontweight='bold')
                    ax_bar.set_title('Probability Distribution - SVM Model', fontsize=14, fontweight='bold', pad=20)
                    ax_bar.set_xlim([0, 1])
                    ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Add percentage labels
                    for i, (bar, prob) in enumerate(zip(bars, prob_df['Probability'])):
                        ax_bar.text(prob + 0.02, i, f'{prob*100:.1f}%', 
                                   va='center', fontweight='bold', fontsize=11)
                    
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close()
                
                else:
                    # Logistic Regression Pipeline Prediction
                    prediction_encoded = pipeline_model.predict([processed_review])[0]
                    probabilities = pipeline_model.predict_proba([processed_review])[0]
                    
                    # Decode prediction from numeric to categorical label
                    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                    
                    # Determine sentiment color and emoji
                    sentiment_config = {
                        'positive': {'color': '#2ecc71', 'gradient': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)', 'emoji': '😊', 'icon': '✅'},
                        'negative': {'color': '#e74c3c', 'gradient': 'linear-gradient(135deg, #eb3349 0%, #f45c43 100%)', 'emoji': '😞', 'icon': '❌'},
                        'neutral': {'color': '#3498db', 'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'emoji': '😐', 'icon': '➖'}
                    }
                    
                    config = sentiment_config.get(prediction.lower(), sentiment_config['neutral'])
                    max_prob = probabilities.max()
                    
                    # Display result in a modern card
                    st.markdown(f"""
                        <div style='background: {config['gradient']}; 
                                    padding: 2rem; border-radius: 15px; text-align: center; 
                                    box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 2rem 0;'>
                            <h2 style='color: white; margin: 0; font-size: 1.2rem; opacity: 0.9;'>🤖 Model: Logistic Regression</h2>
                            <h1 style='color: white; margin: 1rem 0; font-size: 3rem;'>{config['emoji']}</h1>
                            <h2 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>{config['icon']} {prediction.upper()}</h2>
                            <p style='color: white; margin: 1rem 0 0 0; font-size: 1.3rem; opacity: 0.95;'>Confidence: {max_prob*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Get categorical class names from label_encoder
                    class_names = label_encoder.classes_
                    
                    # Show probabilities for each class
                    prob_data = {
                        'Sentimen': class_names,
                        'Probability': probabilities
                    }
                    prob_df = pd.DataFrame(prob_data)
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Modern probability display
                    st.markdown("""
                        <div class='section-container'>
                            <h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📊 Distribusi Probabilitas</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create columns for each sentiment
                    cols = st.columns(len(prob_df))
                    for idx, (col, row) in enumerate(zip(cols, prob_df.itertuples())):
                        sentiment_name = row.Sentimen
                        sentiment_prob = row.Probability
                        sent_config = sentiment_config.get(sentiment_name.lower(), sentiment_config['neutral'])
                        
                        with col:
                            st.markdown(f"""
                                <div style='background: white; padding: 1.5rem; border-radius: 10px; 
                                            box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center;
                                            border-top: 4px solid {sent_config['color']};'>
                                    <h3 style='color: {sent_config['color']}; margin: 0; font-size: 1rem;'>{sent_config['emoji']} {sentiment_name.upper()}</h3>
                                    <p style='font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{sentiment_prob*100:.1f}%</p>
                                    <div style='background: #ecf0f1; border-radius: 10px; height: 8px; margin-top: 0.5rem;'>
                                        <div style='background: {sent_config['color']}; width: {sentiment_prob*100}%; height: 100%; border-radius: 10px;'></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Add bar chart visualization for Logistic Regression
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                    
                    # Prepare colors for each sentiment
                    bar_colors = [sentiment_config.get(sent.lower(), sentiment_config['neutral'])['color'] 
                                  for sent in prob_df['Sentimen']]
                    
                    # Create horizontal bar chart
                    bars = ax_bar.barh(prob_df['Sentimen'], prob_df['Probability'], color=bar_colors, height=0.6)
                    
                    # Styling
                    ax_bar.set_xlabel('Probability', fontsize=12, fontweight='bold')
                    ax_bar.set_title('Probability Distribution - Logistic Regression Model', fontsize=14, fontweight='bold', pad=20)
                    ax_bar.set_xlim([0, 1])
                    ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Add percentage labels
                    for i, (bar, prob) in enumerate(zip(bars, prob_df['Probability'])):
                        ax_bar.text(prob + 0.02, i, f'{prob*100:.1f}%', 
                                   va='center', fontweight='bold', fontsize=11)
                    
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close()
    else:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f5af19 0%, #f12711 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; margin-top: 2rem;'>
                <h2 style='color: white; margin: 0;'>⚠️ Model Tidak Dapat Dimuat!</h2>
                <p style='color: white; margin: 0.5rem 0 0 0; opacity: 0.9;'>Silakan cek kembali model dan data Anda.</p>
            </div>
        """, unsafe_allow_html=True)

# ====================
# PAGE 4: DATA OVERVIEW
# ====================
elif page == "📈 Data Overview":
    st.markdown("""
        <div class='header-box'>
            <h1 style='margin: 0; font-size: 2.5rem;'>📈 Data Overview</h1>
            <p style='margin-top: 0.5rem; font-size: 1.2rem; opacity: 0.9;'>Analisis dan Visualisasi Dataset</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='color: #667eea; margin: 0; font-size: 0.9rem;'>📚 TOTAL REVIEWS</h3>
                    <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{}</p>
                </div>
            """.format(len(df)), unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='color: #667eea; margin: 0; font-size: 0.9rem;'>📊 TOTAL COLUMNS</h3>
                    <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{}</p>
                </div>
            """.format(len(df.columns)), unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='color: #667eea; margin: 0; font-size: 0.9rem;'>⚠️ MISSING VALUES</h3>
                    <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; color: #2c3e50;'>{}</p>
                </div>
            """.format(df.isnull().sum().sum()), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show dataset preview
        st.markdown("""
            <div class='section-container'>
                <h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📋 Dataset Preview</h2>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sentiment distribution
        if 'polarity' in df.columns:
            st.markdown("""
                <div class='section-container'>
                    <h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📊 Distribusi Sentimen</h2>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Count plot
                sentiment_counts = df['polarity'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set2(np.linspace(0, 1, len(sentiment_counts)))
                sentiment_counts.plot(kind='bar', ax=ax, color=colors)
                ax.set_title('Distribusi Sentimen (Count)')
                ax.set_xlabel('Sentimen')
                ax.set_ylabel('Jumlah Review')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set2(np.linspace(0, 1, len(sentiment_counts)))
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
                ax.set_title('Distribusi Sentimen (Persentase)')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Detailed statistics
            st.markdown("<h3 style='color: #2c3e50; margin-top: 2rem;'>📊 Statistik Detail</h3>", unsafe_allow_html=True)
            stats_data = {
                'Sentimen': sentiment_counts.index,
                'Jumlah': sentiment_counts.values,
                'Persentase': (sentiment_counts.values / len(df) * 100).round(2)
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Review length analysis
        st.markdown("""
            <div class='section-container'>
                <h2 style='color: #2c3e50; margin-bottom: 1.5rem;'>📝 Analisis Panjang Review</h2>
        """, unsafe_allow_html=True)
        df['review_length'] = df['content'].fillna('').apply(lambda x: len(str(x).split()))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['review_length'], bins=50, color='#667eea', edgecolor='#764ba2', alpha=0.7)
        ax.set_xlabel('Panjang Review (Jumlah Kata)')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Distribusi Panjang Review')
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Rata-rata Panjang</p>
                    <h3 style='color: white; margin: 0.5rem 0 0 0;'>{:.2f} kata</h3>
                </div>
            """.format(df['review_length'].mean()), unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Min Panjang</p>
                    <h3 style='color: white; margin: 0.5rem 0 0 0;'>{} kata</h3>
                </div>
            """.format(df['review_length'].min()), unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Max Panjang</p>
                    <h3 style='color: white; margin: 0.5rem 0 0 0;'>{} kata</h3>
                </div>
            """.format(df['review_length'].max()), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
       

# ====================
# PAGE 5: UPLOAD CSV (FULL NOTEBOOK CODE)
# ====================
elif page == "📂 Upload CSV":
    st.markdown("""
        <div class='header-box'>
            <h1 style='margin: 0; font-size: 2.5rem;'>📂 Upload & Analisis CSV</h1>
            <p style='margin-top: 0.5rem; font-size: 1.2rem; opacity: 0.9;'>Full Preprocessing & Modeling sesuai Notebook</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.markdown("<h2>📤 Upload File CSV</h2>", unsafe_allow_html=True)
    st.markdown("<p>Upload CSV dengan kolom <code>content</code>. Sistem akan menjalankan full pipeline preprocessing dan modeling seperti di notebook.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="csv_full_notebook")
    
    if uploaded_file is not None:
        try:
            # =======================
            # LOAD DATA
            # =======================
            ajaib_reviews_df = pd.read_csv(uploaded_file)
            
            st.success("✅ File berhasil diupload!")
            
            jumlah_ulasan, jumlah_kolom = ajaib_reviews_df.shape
            st.info(f"📊 Jumlah baris: {jumlah_ulasan} | Kolom: {jumlah_kolom}")
            
            with st.expander("👁️ Preview 5 baris pertama"):
                st.dataframe(ajaib_reviews_df.head(), use_container_width=True)
            
            # Pastikan kolom content ada
            if 'content' not in ajaib_reviews_df.columns:
                st.error("⚠️ Kolom 'content' tidak ditemukan!")
            else:
                # =======================
                # DROP KOLOM TIDAK PERLU
                # =======================
                cols_to_drop = [
                    'reviewId', 'userName', 'userImage', 'score', 'thumbsUpCount', 
                    'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'appVersion'
                ]
                existing_cols = [c for c in cols_to_drop if c in ajaib_reviews_df.columns]
                if existing_cols:
                    ajaib_reviews_df.drop(columns=existing_cols, inplace=True)
                    st.info(f"🗑️ Dropped columns: {', '.join(existing_cols)}")
                
                # Cleaning dasar
                initial_count = len(ajaib_reviews_df)
                ajaib_reviews_df = ajaib_reviews_df.dropna()
                ajaib_reviews_df = ajaib_reviews_df.drop_duplicates()
                cleaned_count = len(ajaib_reviews_df)
                
                st.success(f"✅ Data cleaning: {initial_count} → {cleaned_count} baris")
                
                # =======================
                # PREPROCESSING TEKS
                # =======================
                with st.spinner("⚙️ Running full preprocessing pipeline..."):
                    ajaib_reviews_df['cleaned_text'] = ajaib_reviews_df['content'].apply(remove_noise)
                    ajaib_reviews_df['lowercased_text'] = ajaib_reviews_df['cleaned_text'].apply(to_lowercase)
                    ajaib_reviews_df['normalized_text'] = ajaib_reviews_df['lowercased_text'].apply(normalize_slang_notebook)
                    ajaib_reviews_df['tokens'] = ajaib_reviews_df['normalized_text'].apply(tokenize_words)
                    ajaib_reviews_df['filtered_tokens'] = ajaib_reviews_df['tokens'].apply(remove_stopwords)
                    ajaib_reviews_df['final_text'] = ajaib_reviews_df['filtered_tokens'].apply(reconstruct_text)
                
                st.success("✅ Preprocessing complete!")
                
                with st.expander("🔍 Sample Result"):
                    st.dataframe(ajaib_reviews_df.head(2), use_container_width=True)
                
                # =======================
                # PELABELAN LEXICON
                # =======================
                st.markdown("### 🏷️ Lexicon-Based Labeling")
                
                with st.spinner("📚 Loading lexicon..."):
                    import csv as _csv
                    from io import StringIO
                    
                    lexicon_positive = {}
                    try:
                        rpos = requests.get(
                            'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv',
                            timeout=10
                        )
                        rpos.raise_for_status()
                        reader = _csv.reader(StringIO(rpos.text))
                        for row in reader:
                            if len(row) >= 2:
                                lexicon_positive[row[0]] = int(row[1])
                    except Exception as e:
                        st.warning(f"Failed to load positive lexicon: {e}")
                    
                    lexicon_negative = {}
                    try:
                        rneg = requests.get(
                            'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv',
                            timeout=10
                        )
                        rneg.raise_for_status()
                        reader = _csv.reader(StringIO(rneg.text))
                        for row in reader:
                            if len(row) >= 2:
                                lexicon_negative[row[0]] = int(row[1])
                    except Exception as e:
                        st.warning(f"Failed to load negative lexicon: {e}")
                    
                    def sentiment_analysis_lexicon_indonesia(text):
                        score = 0
                        for word in text:
                            if word in lexicon_positive:
                                score += lexicon_positive[word]
                        for word in text:
                            if word in lexicon_negative:
                                score += lexicon_negative[word]
                        
                        if score > 0:
                            polarity = 'positive'
                        elif score < 0:
                            polarity = 'negative'
                        else:
                            polarity = 'neutral'
                        return score, polarity
                    
                    results = ajaib_reviews_df['filtered_tokens'].apply(sentiment_analysis_lexicon_indonesia)
                    results = list(zip(*results))
                    ajaib_reviews_df['polarity_score'] = results[0]
                    ajaib_reviews_df['polarity'] = results[1]
                
                st.success("✅ Labeling complete!")
                
                # Distribusi sentimen
                polarity_counts = ajaib_reviews_df['polarity'].value_counts()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(
                        pd.DataFrame({
                            'Sentiment': polarity_counts.index,
                            'Count': polarity_counts.values,
                            'Percentage': (polarity_counts.values / len(ajaib_reviews_df) * 100).round(2)
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors_map = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
                    plot_colors = [colors_map.get(x, '#95a5a6') for x in polarity_counts.index]
                    ax.pie(
                        polarity_counts.values,
                        labels=polarity_counts.index,
                        autopct='%1.1f%%',
                        colors=plot_colors,
                        startangle=90
                    )
                    ax.set_title('Sentiment Distribution')
                    st.pyplot(fig)
                    plt.close()
                
                # =======================
                # MODELING
                # =======================
                st.markdown("### 🤖 Machine Learning Modeling")
                
                with st.spinner("🔄 Training models..."):
                    from sklearn.model_selection import train_test_split, GridSearchCV
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.svm import LinearSVC
                    from sklearn.calibration import CalibratedClassifierCV
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.feature_selection import SelectKBest, chi2
                    from sklearn.decomposition import TruncatedSVD
                    from sklearn.pipeline import Pipeline
                    from sklearn.metrics import accuracy_score, confusion_matrix
                    import numpy as np
                    
                    # --- data untuk model ---
                    X = ajaib_reviews_df['final_text'].fillna("").astype(str)
                    y = ajaib_reviews_df['polarity']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=0.2,
                        random_state=42,
                        stratify=y if len(y.unique()) > 1 else None
                    )
                    
                    le = LabelEncoder()
                    y_train_enc = le.fit_transform(y_train)
                    y_test_enc = le.transform(y_test)
                    
                    # ============================
                    # 1) SVM + TF-IDF (BUFFED)
                    # ============================
                    svm_pipe = Pipeline([
                        ("tfidf", TfidfVectorizer(
                            max_features=12000,      # banyak fitur → kaya informasi
                            ngram_range=(1, 3),      # unigram + bigram + trigram
                            analyzer="word",
                            lowercase=True,
                            min_df=2,
                            max_df=0.98,
                            sublinear_tf=True
                        )),
                        ("clf", CalibratedClassifierCV(
                            estimator=LinearSVC(),
                            cv=3
                        ))
                    ])
                    
                    # CV: perhatikan jumlah data per kelas
                    class_counts = np.bincount(y_train_enc)
                    min_class = class_counts.min() if len(class_counts) > 0 else 2
                    cv_splits = int(max(2, min(5, min_class)))
                    
                    svm_param_grid = {
                        "clf__estimator__C": [0.1, 1, 2, 5, 10],
                        "clf__estimator__class_weight": [None, "balanced"]
                    }
                    
                    svm_gs = GridSearchCV(
                        svm_pipe,
                        svm_param_grid,
                        cv=cv_splits,
                        scoring="f1_weighted",
                        n_jobs=-1,
                        verbose=0
                    )
                    svm_gs.fit(X_train, y_train_enc)
                    
                    svm_model_upload = svm_gs.best_estimator_
                    y_pred_svm = svm_model_upload.predict(X_test)
                    svm_accuracy = accuracy_score(y_test_enc, y_pred_svm)
                    
                    st.info(f"🎯 Best SVM params: {svm_gs.best_params_}")
                    
                    # ==============================================
                    # 2) Logistic Regression(TF-IDF + SelectKBest + SVD + LR)
                    # ==============================================
                    temp_vect = TfidfVectorizer(
                        max_features=3000,
                        ngram_range=(1, 2),
                        min_df=2
                    )
                    temp_X = temp_vect.fit_transform(X_train)
                    n_features = temp_X.shape[1]
                    
                    st.info(f"📐 Dataset features (for pipeline): {n_features}")
                    
                    def safe_k(target):
                        return max(2, min(n_features - 1, target))
                    
                    def safe_svd(target):
                        return max(2, min(n_features - 1, target))
                    
                    if n_features < 100:
                        k_options = [
                            safe_k(n_features // 2),
                            safe_k(min(100, n_features - 1))
                        ]
                        svd_options = [
                            safe_svd(max(2, n_features // 3)),
                            safe_svd(max(2, n_features // 2))
                        ]
                    elif n_features < 500:
                        k_options = [
                            safe_k(n_features // 2),
                            safe_k(min(200, n_features - 1))
                        ]
                        svd_options = [
                            safe_svd(50),
                            safe_svd(100)
                        ]
                    else:
                        k_options = [safe_k(800), safe_k(1500)]
                        svd_options = [safe_svd(150), safe_svd(300)]
                    
                    vectorizer = TfidfVectorizer(
                        max_features=3000,
                        ngram_range=(1, 2),
                        min_df=2
                    )
                    select_k = SelectKBest(chi2, k=k_options[0])
                    svd = TruncatedSVD(n_components=svd_options[0], random_state=42)
                    clf = LogisticRegression(
                        solver='saga',
                        max_iter=2000,
                        C=1.0,
                        n_jobs=-1
                    )
                    
                    pipe = Pipeline([
                        ("vect", vectorizer),
                        ("select", select_k),
                        ("svd", svd),
                        ("clf", clf)
                    ])
                    
                    param_grid = {
                        "select__k": k_options,
                        "svd__n_components": svd_options,
                        "clf__C": [0.5, 1.0]
                    }
                    
                    gs = GridSearchCV(
                        pipe,
                        param_grid,
                        cv=cv_splits,
                        scoring="f1_weighted",
                        n_jobs=-1,
                        verbose=0
                    )
                    gs.fit(X_train, y_train_enc)
                    
                    best_model_upload = gs.best_estimator_
                    y_pred_pipe = best_model_upload.predict(X_test)
                    pipe_accuracy = accuracy_score(y_test_enc, y_pred_pipe)
                
                st.success("✅ Models trained!")
                
                # =======================
                # HASIL MODEL
                # =======================
                st.markdown("### 📊 Model Performance")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; border-radius: 15px; text-align: center;'>
                            <h4 style='color: white; margin: 0;'>SVM + TF-IDF</h4>
                            <h2 style='color: white; margin: 0.5rem 0 0 0; font-size: 2.5rem;'>{svm_accuracy:.2%}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%); 
                                    padding: 1.5rem; border-radius: 15px; text-align: center;'>
                            <h4 style='color: white; margin: 0;'>Logistic Regression</h4>
                            <h2 style='color: white; margin: 0.5rem 0 0 0; font-size: 2.5rem;'>{pipe_accuracy:.2%}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                if pipe_accuracy > svm_accuracy:
                    st.success(f"✅ Logistic Regression better by {(pipe_accuracy - svm_accuracy):.4f}")
                else:
                    st.success(f"✅ SVM better by {(svm_accuracy - pipe_accuracy):.4f}")
                
                # =======================
                # CONFUSION MATRIX
                # =======================
                st.markdown("### 🎯 Confusion Matrices")
                cc1, cc2 = st.columns(2)
                
                with cc1:
                    cm_svm = confusion_matrix(y_test_enc, y_pred_svm)
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        cm_svm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=le.classes_,
                        yticklabels=le.classes_,
                        ax=ax1
                    )
                    ax1.set_title("SVM")
                    st.pyplot(fig1)
                    plt.close()
                
                with cc2:
                    cm_pipe = confusion_matrix(y_test_enc, y_pred_pipe)
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        cm_pipe,
                        annot=True,
                        fmt='d',
                        cmap='Greens',
                        xticklabels=le.classes_,
                        yticklabels=le.classes_,
                        ax=ax2
                    )
                    ax2.set_title("Logistic Regression")
                    st.pyplot(fig2)
                    plt.close()
                
                # =======================
                # DOWNLOAD HASIL (Hanya Data yang Sudah Dilabeli)
                # =======================
                st.markdown("### 📥 Download Hasil")
                
                # Pilih kolom penting untuk download: content, hasil preprocessing, dan label polarity
                download_columns = ['content', 'cleaned_text', 'lowercased_text', 'normalized_text', 
                                   'final_text', 'polarity_score', 'polarity']
                
                # Filter kolom yang ada
                available_cols = [col for col in download_columns if col in ajaib_reviews_df.columns]
                df_download = ajaib_reviews_df[available_cols]
                
                st.info(f"📊 Total data yang sudah dilabeli: {len(df_download)} baris")
                
                with st.expander("👁️ Preview data yang akan didownload (5 baris pertama)"):
                    st.dataframe(df_download.head(), use_container_width=True)
                
                csv_output = df_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data yang Sudah Dilabeli (CSV)",
                    data=csv_output,
                    file_name="data_labeled_hasil.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.error(traceback.format_exc())
    
    st.markdown("</div>", unsafe_allow_html=True)
