import os
import cv2
import numpy as np
import streamlit as st
from random import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import pandas as pd
from rembg import remove
from PIL import Image

USER_CREDENTIALS = {
    "admin": "1234"
}

#PAGE TITLE AND SET
st.set_page_config(
    page_title="Advanced ML Classification Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

##CSS FOR STYLING
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    .authors-box {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(44, 62, 80, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .authors-box h3 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .authors-names {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    .model-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 5px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(52, 152, 219, 0.3), transparent);
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    .model-card.decision-tree {
        border-left-color: #3498db;
    }
    
    .model-card.naive-bayes {
        border-left-color: #2ecc71;
    }
    
    .model-card.cnn-mlp {
        border-left-color: #e74c3c;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .run-results {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
    }
    
    .run-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .accuracy-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    
    .accuracy-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .accuracy-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .accuracy-label {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    .tree-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin: 2rem 0;
        border: 1px solid #ecf0f1;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db, #2ecc71);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
    }
    
    .status-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-info {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        display: inline-block;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #2ecc71;
    }

    /* Beautiful Centered Login Container */
    .login-page {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }

    .login-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        width: 100%;
        max-width: 630px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .login-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .login-subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    .stTextInput > div > div > input {
        padding: 1rem !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        border: 2px solid #e9ecef !important;
        transition: all 0.3s ease !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1) !important;
        outline: none !important;
    }
    
    .stTextInput > label {
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.5rem !important;
        font-size: 1rem !important;
    }
    
    .login-button {
        margin-top: 1rem;
    }
    
    .login-button button {
        width: 100% !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%) !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 20px rgba(52, 152, 219, 0.3) !important;
    }
    
    .login-button button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(52, 152, 219, 0.4) !important;
    }
    
    .login-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
</style>
""", unsafe_allow_html=True)

#DATA LOC
DATA_DIR = "Output"
CLASSES = ["banana", "blueberries", "pomegranates"]
IMAGE_SIZE = (64, 64)
BLOCKS = (4, 4)
## REMOVES THE BACKGROUND USING AI FOR UPLOADED PICS
def remove_bg_ai(image_bgr):
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb).convert("RGBA")

        output_pil = remove(pil_image)
        output_rgba = np.array(output_pil)

        alpha = output_rgba[:, :, 3]

        result_rgba = cv2.cvtColor(output_rgba, cv2.COLOR_RGBA2BGRA)

        return result_rgba, alpha

    except Exception as e:
        st.error(f"❌ Error during background removal: {e}")
        fallback = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        return fallback, np.ones((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8) * 255

##THIS WILL BE USED FOR THE DT
def preprocess_image_blocked_rgb(image, blocks=BLOCKS):
    image = cv2.resize(image, IMAGE_SIZE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    white_mask = np.all(image_rgb > 0.9, axis=2)
    image_rgb[white_mask] = 0.0
    h_blocks, w_blocks = blocks
    h_step = IMAGE_SIZE[0] // h_blocks
    w_step = IMAGE_SIZE[1] // w_blocks
    features = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = image_rgb[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            mean_vals = block.mean(axis=(0, 1))
            features.extend(mean_vals)
    return (np.array(features) * 255).astype(np.uint8)


## THIS GETS THE DATAS MEAN AND MORE SO FOR THE NB
def extract_statistical_features(image):
    image = cv2.resize(image, IMAGE_SIZE)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    features = []
    features.append(np.mean(gray))
    features.append(np.std(gray))
    
    if len(image.shape) == 3:
        for i in range(3):
            channel = image[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
    else:
        for i in range(3):
            features.append(np.mean(gray))
            features.append(np.std(gray))
    
    return np.array(features)

def get_cnn_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1], 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def extract_cnn_features(model, image):
    image = cv2.resize(image,IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(np.expand_dims(image.astype(np.float32), axis=0))
    features = model.predict(image, verbose=0)
    return features.flatten()


##FOR THE UPLOADED IMAGE WE NEED TO PROCESS IT LIKE THE OTHER ONES SO IT CAN BE UNBIASED
def process_uploaded_image_for_prediction(uploaded_file, cnn_model=None):
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # Ensure we have a valid image
    if img_bgr is None:
        st.error("❌ Could not decode image. Please try a different image.")
        return None, None
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    #REM BACKGROUND
    masked_img, final_mask = remove_bg_ai(img_bgr)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    
    #DISPLAY THE PIC
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.image(img_rgb, caption="Original Image", use_container_width=True)
    with col3:
        pass
    
    ##FEATURE EXTRACTION-----
    features = {}
    features['dt'] = preprocess_image_blocked_rgb(masked_img)
    features['nb'] = extract_statistical_features(masked_img)
    
    if cnn_model is not None:
        features['cnn'] = extract_cnn_features(cnn_model, masked_img)
    
    return img_rgb, features
##LOAD THE DATA FROM THE FOLDER
def load_image_paths_and_labels():
    """Load and balance dataset"""
    X_paths, y, paths = [], [], []
    label_map = {name: idx for idx, name in enumerate(CLASSES)}
    

    
    class_files = {}
    
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.exists(folder):
            st.error(f"❌ Data folder '{folder}' not found!")
            return [], np.array([]), {}
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_files[cls] = sorted(files)
    
    min_count = min(len(files) for files in class_files.values())
    
    st.markdown(f"""
    <div class="status-info">
        <strong>📊 Dataset Information:</strong><br>
        Balanced dataset by minimum class size: <strong>{min_count}</strong> images per class.<br>
        <strong>Note:</strong> Training data is preprocessed with background removal for optimal performance.
    </div>
    """, unsafe_allow_html=True)
    
    for cls in CLASSES:
        st.write(f"- **{cls}**: {min_count} images")
    
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        for fname in class_files[cls][:min_count]:
            img_path = os.path.join(folder, fname)
            y.append(label_map[cls])
            paths.append(img_path)
    
    combined = list(zip(paths, y))
    shuffle(combined)
    paths, y = zip(*combined)
    return list(paths), np.array(y), label_map


##CREATE TREE VISUALSSSS
def create_tree_graphviz(model, class_names):
    feature_names = []
    for b in range(BLOCKS[0]*BLOCKS[1]):
        feature_names.extend([f'block_{b}_R', f'block_{b}_G', f'block_{b}_B'])

    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True,
        impurity=False,
        label='all',
    )

    dot_data = dot_data.replace(
        "digraph Tree {", 
        """digraph Tree {
        rankdir=TB;
        node [fontname="Inter", fontsize=12, style="filled,rounded", shape=box, margin=0.1];
        edge [fontname="Inter", fontsize=10, color="#34495e"];
        bgcolor="transparent";
        """
    )

    emoji_map = {
        "banana": "🍌",
        "blueberries": "🫐", 
        "pomegranates": "🍎"
    }
    
    for cname, emoji in emoji_map.items():
        dot_data = dot_data.replace(f'class = "{cname}"', f'class = "{cname} {emoji}"')
        dot_data = dot_data.replace(f"class = {cname}", f"class = {cname} {emoji}")

    return graphviz.Source(dot_data)


##CREATE THE REPORTS
def create_classification_report_table(y_true, y_pred, class_names, model_name):
    """Create classification report as DataFrame"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    df_data = []
    for class_name in class_names:
        if class_name in report:
            df_data.append({
                'Class': class_name.title(),
                'Precision': round(report[class_name]['precision'], 3),
                'Recall': round(report[class_name]['recall'], 3),
                'F1-Score': round(report[class_name]['f1-score'], 3),
                'Support': int(report[class_name]['support'])
            })
    
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            df_data.append({
                'Class': avg_type.title(),
                'Precision': round(report[avg_type]['precision'], 3),
                'Recall': round(report[avg_type]['recall'], 3),
                'F1-Score': round(report[avg_type]['f1-score'], 3),
                'Support': int(report[avg_type]['support'])
            })
    
    df = pd.DataFrame(df_data)
    return df


##PLOT CONF MATRIX
def plot_confusion_matrix(ax, cm, class_names, title):
    """Plot confusion matrix"""
    cmap = plt.cm.Blues
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.8,
        linecolor='gray',
        annot_kws={"size":14, "weight":'bold'},
        vmin=0,
        vmax=cm.max(),
    )
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    for text in ax.texts:
        val = float(text.get_text())
        threshold = cm.max() / 2.
        text.set_color("white" if val > threshold else "black")

def predict_single_image(image_features, models, scalers, class_names):
    """Make predictions on a single image using all trained models"""
    predictions = {}
    
    if 'dt_model' in models and models['dt_model'] is not None:
        dt_pred = models['dt_model'].predict(image_features['dt'].reshape(1, -1))[0]
        predictions['Decision Tree'] = class_names[dt_pred]

    
    if 'nb_model' in models and models['nb_model'] is not None:
        nb_feat_scaled = scalers['nb'].transform(image_features['nb'].reshape(1, -1))
        nb_pred = models['nb_model'].predict(nb_feat_scaled)[0]
        predictions['Naive Bayes'] = class_names[nb_pred]

    
    if 'mlp_model' in models and models['mlp_model'] is not None:
        cnn_feat_scaled = scalers['cnn'].transform(image_features['cnn'].reshape(1, -1))
        mlp_pred = models['mlp_model'].predict(cnn_feat_scaled)[0]
        predictions['CNN + MLP'] = class_names[mlp_pred]

    
    return predictions


##MAIN WEBPAGE
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Advanced ML Classification Suite</h1>
        <p>Comparative Study of Image Classification Using Decision Tree, Naive Bayes, and CNN+MLP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'experiment_completed' not in st.session_state:
        st.session_state.experiment_completed = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'trained_scalers' not in st.session_state:
        st.session_state.trained_scalers = {}
    if 'dt_models' not in st.session_state:
        st.session_state.dt_models = []
    if 'class_names' not in st.session_state:
        st.session_state.class_names = []
    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = None
    if 'last_predictions' not in st.session_state:
        st.session_state.last_predictions = {}
    
    #SIDEBAR CONFIG
    st.sidebar.markdown("## ⚙️ Configuration Panel")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🤖 Select Models to Run")
    run_decision_tree = st.sidebar.checkbox("🌳 Decision Tree", value=True)
    run_naive_bayes = st.sidebar.checkbox("⚡ Naive Bayes", value=True)
    run_cnn_mlp = st.sidebar.checkbox("🧠 CNN + MLP", value=True)
    
    if not any([run_decision_tree, run_naive_bayes, run_cnn_mlp]):
        st.sidebar.error("⚠️ Please select at least one model to run!")
        return
    
    st.sidebar.markdown("### 🔄 Cross-Validation Settings")
    N_RUNS = st.sidebar.slider("Number of Runs", min_value=1, max_value=10, value=5, step=1)
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    
    st.sidebar.markdown("### 🔧 Advanced Settings")
    random_state = st.sidebar.number_input("Random State", min_value=1, max_value=100, value=42)
    
  # DECISION TREE PRUNE
    st.sidebar.markdown("#### 🌳 Decision Tree Pruning")
    max_depth = st.sidebar.slider("Max Depth", min_value=2, max_value=20, value=10, step=1)
    min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=50, value=10, step=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=20, value=5, step=1)
    min_impurity_decrease = st.sidebar.slider("Min Impurity Decrease", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    user_ccp_alpha = st.sidebar.slider("Post-Pruning Alpha (ccp_alpha)", min_value=0.0, max_value=0.05, value=0.005, step=0.001,format="%.3f")

    # PRUNE INFO BOX
    st.sidebar.markdown("""
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-size: 0.8rem;">
        <strong>🔍 Pruning for Dataset:</strong><br>
        • <strong>Max Depth (10):</strong> Safe limit, other factors should converge before<br>
        • <strong>Min Samples Split (10):</strong> Prevents splitting small nodes<br>
        • <strong>Min Samples Leaf (5):</strong> Each leaf must have ≥5 samples<br>
        • <strong>Min Impurity Decrease (0.01):</strong> Split must improve purity by ≥1%<br>
        • <strong>Post-Pruning Alpha:</strong> Higher alpha prunes more aggressively<br>
        </div>
        """, unsafe_allow_html=True)

    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Project Report")

    try:
        with open("AiReport.pdf", "rb") as f:
            pdf_data = f.read()
            st.sidebar.download_button(
                label="📥 Download Final Report",
                data=pdf_data,
                file_name="AiReport.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.sidebar.error("❌ 'AiReport.pdf' not found.")
    
    #AUTHOURS BOX
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="authors-box">
        <h3>👥 Project Authors</h3>
        <div class="authors-names">Heba Mustafa & Mohammad Omar</div>
    </div>
    """, unsafe_allow_html=True)
    
    #MODEL INFORMATION CARDS
    st.markdown('<div class="section-header">🤖 Model Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if run_decision_tree:
            st.markdown("""
            <div class="model-card decision-tree">
                <h3>🌳 Decision Tree</h3>
                <p><strong>Feature Extraction:</strong> Color Pixel Intensity Features</p>
                <p><strong>Method:</strong> Rule-based classification with interpretable paths</p>
                <p><strong>Advantages:</strong> Interpretable, fast training, visual decision paths</p>
                <p><strong>Status:</strong> <span style="color: #27ae60;">✅ Selected</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-card" style="opacity: 0.5;">
                <h3>🌳 Decision Tree</h3>
                <p><strong>Status:</strong> <span style="color: #e74c3c;">❌ Not Selected</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if run_naive_bayes:
            st.markdown("""
            <div class="model-card naive-bayes">
                <h3>⚡ Naive Bayes</h3>
                <p><strong>Feature Extraction:</strong> Statistical features (mean, std)</p>
                <p><strong>Method:</strong> Probabilistic classification</p>
                <p><strong>Advantages:</strong> Fast inference, robust to noise, baseline model</p>
                <p><strong>Status:</strong> <span style="color: #27ae60;">✅ Selected</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-card" style="opacity: 0.5;">
                <h3>⚡ Naive Bayes</h3>
                <p><strong>Status:</strong> <span style="color: #e74c3c;">❌ Not Selected</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if run_cnn_mlp:
            st.markdown("""
            <div class="model-card cnn-mlp">
                <h3>🧠 CNN + MLP</h3>
                <p><strong>Feature Extraction:</strong> VGG16 deep features</p>
                <p><strong>Method:</strong> Transfer learning + MLP classifier</p>
                <p><strong>Advantages:</strong> High accuracy, rich feature representation</p>
                <p><strong>Status:</strong> <span style="color: #27ae60;">✅ Selected</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-card" style="opacity: 0.5;">
                <h3>🧠 CNN + MLP</h3>
                <p><strong>Status:</strong> <span style="color: #e74c3c;">❌ Not Selected</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    #UPLOAD THE IMAGE FOR THE TESTINF
    st.markdown('<div class="section-header">📸 Upload Image to Test During Runs</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], key="upload_input")
    
    if uploaded_image:
        if run_cnn_mlp and st.session_state.cnn_model is None:
            with st.spinner("Loading CNN feature extractor..."):
                st.session_state.cnn_model = get_cnn_feature_extractor()
    
        #PROCESS
        uploaded_image_copy = uploaded_image
        img_rgb, upload_features = process_uploaded_image_for_prediction(
            uploaded_image_copy, 
            st.session_state.cnn_model if run_cnn_mlp else None
        )
    
        if img_rgb is not None and upload_features is not None:
            st.session_state.upload_features = upload_features
            st.session_state.upload_predictions = {'dt': [], 'nb': [], 'mlp': []}
        
    
    #START 
    st.markdown('<div class="section-header">🚀 Run Experiment</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        start_experiment = st.button("🎯 Start Classification Experiment", key="start_experiment")
    with col2:
        if st.button("🔄 Reset", key="reset_experiment"):
            for key in ['experiment_completed', 'results', 'trained_models', 'trained_scalers', 'last_predictions',
                       'dt_models', 'class_names', 'cnn_model']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    if start_experiment:
        #LOAD DATA AND GET READY 
        paths, y, label_map = load_image_paths_and_labels()
        if len(paths) == 0:
            st.error("❌ No data found! Please check your data directory.")
            return
            
        st.session_state.class_names = list(label_map.keys())
        
        #TRACKER
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        #EXTRACTING..........
        features_dict = {}
        total_images = len(paths)
        
        if run_decision_tree:
            status_text.markdown('<div class="status-info">📊 Extracting features for Decision Tree...</div>', unsafe_allow_html=True)
            dt_features = []
            for i, p in enumerate(paths):
                img = cv2.imread(p)
                if img is not None:
                    dt_features.append(preprocess_image_blocked_rgb(img))
                progress_bar.progress((i + 1) / total_images * 0.3)
            features_dict['dt'] = np.array(dt_features)

        if run_naive_bayes:
            status_text.markdown('<div class="status-info">📊 Extracting features for Naive Bayes...</div>', unsafe_allow_html=True)
            nb_features = []
            for i, p in enumerate(paths):
                img = cv2.imread(p)
                if img is not None:
                    nb_features.append(extract_statistical_features(img))
                progress_bar.progress(0.3 + (i + 1) / total_images * 0.3)
            features_dict['nb'] = np.array(nb_features)

        if run_cnn_mlp:
            status_text.markdown('<div class="status-info">🧠 Loading CNN feature extractor (VGG16)...</div>', unsafe_allow_html=True)
            if st.session_state.cnn_model is None:
                st.session_state.cnn_model = get_cnn_feature_extractor()
            
            status_text.markdown('<div class="status-info">🔍 Extracting CNN features...</div>', unsafe_allow_html=True)
            cnn_features = []
            for i, p in enumerate(paths):
                img = cv2.imread(p)
                if img is not None:
                    cnn_features.append(extract_cnn_features(st.session_state.cnn_model, img))
                progress_bar.progress(0.6 + (i + 1) / total_images * 0.4)
            features_dict['cnn'] = np.array(cnn_features)
        
        progress_bar.progress(1.0)
        status_text.markdown('<div class="status-success">✅ Feature extraction completed!</div>', unsafe_allow_html=True)
        
        #STORE 
        st.session_state.results = {}
        st.session_state.last_predictions = {}
        st.session_state.dt_models = []
        
        if run_decision_tree:
            st.session_state.results['dt'] = {'accuracies': [], 'conf_matrices': []}
        if run_naive_bayes:
            st.session_state.results['nb'] = {'accuracies': [], 'conf_matrices': []}
        if run_cnn_mlp:
            st.session_state.results['mlp'] = {'accuracies': [], 'conf_matrices': []}
        
        # Cross-validation
        st.markdown('<div class="section-header">🔄 Cross-Validation Results</div>', unsafe_allow_html=True)

        st.markdown("""
<div class="status-info">
    <strong>🔄 Cross-Validation Strategy:</strong><br>
    • <strong>Same Splits:</strong> All models use identical train/test splits for each run<br>
    • <strong>Fair Comparison:</strong> Ensures performance differences are due to model, not data<br>
    • <strong>Stratified:</strong> Maintains class distribution in each split<br>
    • <strong>Shuffled:</strong> Different random splits for each run<br>
</div>
""", unsafe_allow_html=True)
        
        splitter = StratifiedShuffleSplit(n_splits=N_RUNS, test_size=test_size, random_state=random_state)
        run_i = 1
        
        for train_idx, test_idx in splitter.split(list(features_dict.values())[0], y):
            y_train, y_test = y[train_idx], y[test_idx]
            run_results = []
            
            # Decision Tree
            if run_decision_tree:
                X_train_dt, X_test_dt = features_dict['dt'][train_idx], features_dict['dt'][test_idx]
                dt_model = DecisionTreeClassifier(
                    random_state=random_state, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_impurity_decrease=min_impurity_decrease,
                    ccp_alpha=user_ccp_alpha
                )
                dt_model.fit(X_train_dt, y_train)
                st.session_state.dt_models.append(dt_model)
                y_pred_dt = dt_model.predict(X_test_dt)
                acc_dt = accuracy_score(y_test, y_pred_dt)
                st.session_state.results['dt']['accuracies'].append(acc_dt)
                st.session_state.results['dt']['conf_matrices'].append(confusion_matrix(y_test, y_pred_dt))
                run_results.append(("🌳 Decision Tree", acc_dt, "#3498db"))
                st.session_state.last_predictions.setdefault('dt', []).append((y_test, y_pred_dt))
                #STORE FINAL 
                if run_i == N_RUNS:
                    st.session_state.trained_models['dt_model'] = dt_model
                    
                
                #PREDICT
                if uploaded_image and 'upload_features' in st.session_state:
                    pred = dt_model.predict(st.session_state.upload_features['dt'].reshape(1, -1))[0]
                    st.session_state.upload_predictions['dt'].append(pred)
            
            # Naive Bayes
            if run_naive_bayes:
                X_train_nb, X_test_nb = features_dict['nb'][train_idx], features_dict['nb'][test_idx]
                scaler_nb = StandardScaler()
                X_train_nb_scaled = scaler_nb.fit_transform(X_train_nb)
                X_test_nb_scaled = scaler_nb.transform(X_test_nb)
                
                nb_model = GaussianNB()
                nb_model.fit(X_train_nb_scaled, y_train)
                y_pred_nb = nb_model.predict(X_test_nb_scaled)
                acc_nb = accuracy_score(y_test, y_pred_nb)
                st.session_state.results['nb']['accuracies'].append(acc_nb)
                st.session_state.results['nb']['conf_matrices'].append(confusion_matrix(y_test, y_pred_nb))
                run_results.append(("⚡ Naive Bayes", acc_nb, "#2ecc71"))
                st.session_state.last_predictions.setdefault('nb', []).append((y_test, y_pred_nb))
                #STORE FINAL 
                if run_i == N_RUNS:
                    st.session_state.trained_models['nb_model'] = nb_model
                    st.session_state.trained_scalers['nb'] = scaler_nb
                
                #PREDICT
                if uploaded_image and 'upload_features' in st.session_state:
                    nb_feat_scaled = scaler_nb.transform(st.session_state.upload_features['nb'].reshape(1, -1))
                    pred = nb_model.predict(nb_feat_scaled)[0]
                    st.session_state.upload_predictions['nb'].append(pred)
            
            # CNN + MLP
            if run_cnn_mlp:
                X_train_cnn, X_test_cnn = features_dict['cnn'][train_idx], features_dict['cnn'][test_idx]
                scaler_cnn = StandardScaler()
                X_train_cnn_scaled = scaler_cnn.fit_transform(X_train_cnn)
                X_test_cnn_scaled = scaler_cnn.transform(X_test_cnn)
                
                mlp_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=random_state)
                mlp_model.fit(X_train_cnn_scaled, y_train)
                y_pred_mlp = mlp_model.predict(X_test_cnn_scaled)
                acc_mlp = accuracy_score(y_test, y_pred_mlp)
                st.session_state.results['mlp']['accuracies'].append(acc_mlp)
                st.session_state.results['mlp']['conf_matrices'].append(confusion_matrix(y_test, y_pred_mlp))
                run_results.append(("🧠 CNN+MLP", acc_mlp, "#e74c3c"))
                st.session_state.last_predictions.setdefault('mlp', []).append((y_test, y_pred_mlp))
                
             
                if run_i == N_RUNS:
                    st.session_state.trained_models['mlp_model'] = mlp_model
                    st.session_state.trained_scalers['cnn'] = scaler_cnn
                    st.session_state.last_predictions.setdefault('mlp', []).append((y_test, y_pred_nb))
                
    
                if uploaded_image and 'upload_features' in st.session_state:
                    cnn_feat_scaled = scaler_cnn.transform(st.session_state.upload_features['cnn'].reshape(1, -1))
                    pred = mlp_model.predict(cnn_feat_scaled)[0]
                    st.session_state.upload_predictions['mlp'].append(pred)
            
            #DISPLAY RESULTSSSSSS
            st.markdown(f"""
            <div class="run-results">
                <div class="run-title">📊 Run {run_i}</div>
                <div class="accuracy-grid">
            """, unsafe_allow_html=True)
            
            for model_name, accuracy, color in run_results:
                st.markdown(f"""
                    <div class="accuracy-item">
                        <div class="accuracy-value" style="color: {color};">{accuracy:.1%}</div>
                        <div class="accuracy-label">{model_name}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            run_i += 1
        

        if uploaded_image and any(st.session_state.upload_predictions.values()):
            st.markdown('<div class="section-header">🔍 Uploaded Image Prediction Summary</div>', unsafe_allow_html=True)
            
            prediction_results = []
            for model_key, preds in st.session_state.upload_predictions.items():
                if preds:
                    model_names = {'dt': '🌳 Decision Tree', 'nb': '⚡ Naive Bayes', 'mlp': '🧠 CNN + MLP'}
                    values, counts = np.unique(preds, return_counts=True)
                    most_common_idx = np.argmax(counts)
                    most_common_class = CLASSES[values[most_common_idx]]
                    confidence = (counts[most_common_idx] / len(preds)) * 100
                    
                    prediction_results.append({
                        "Model": model_names[model_key],
                        "Most Predicted Class": most_common_class,
                        "Confidence": f"{confidence:.1f}%",
                        "All Predictions": ", ".join([CLASSES[p] for p in preds])
                    })
            
            if prediction_results:
                df_preds = pd.DataFrame(prediction_results)
                st.dataframe(df_preds, use_container_width=True)
        
        st.session_state.experiment_completed = True
    

    if st.session_state.experiment_completed:
        st.markdown('<div class="section-header">📈 Final Results Summary</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(st.session_state.results))
        model_names = []
        avg_accuracies = []
        
        for i, (model_key, model_data) in enumerate(st.session_state.results.items()):
            avg_acc = np.mean(model_data['accuracies'])
            avg_accuracies.append(avg_acc)
            
            with cols[i]:
                if model_key == 'dt':
                    model_name = "Decision Tree"
                    icon = "🌳"
                    color = "#3498db"
                elif model_key == 'nb':
                    model_name = "Naive Bayes"
                    icon = "⚡"
                    color = "#2ecc71"
                else:
                    model_name = "CNN + MLP"
                    icon = "🧠"
                    color = "#e74c3c"
                
                model_names.append(model_name)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div class="metric-value" style="color: {color};">{avg_acc*100:.2f}%</div>
                    <div class="metric-label">{model_name}</div>
                </div>
                """, unsafe_allow_html=True)
          #PUT THE REPORTS AS TABLES 
        st.markdown('<div class="section-header">📄 Classification Reports</div>', unsafe_allow_html=True)

        
        for model_key, runs in st.session_state.last_predictions.items():
                st.markdown(f"#### {'🌳 Decision Tree' if model_key == 'dt' else '⚡ Naive Bayes' if model_key == 'nb' else '🧠 CNN + MLP'}")

                for i, (y_test, y_pred) in enumerate(runs, start=1):
                    st.markdown(f"**Run {i}**")

                    report_dict = classification_report(y_test, y_pred, target_names=st.session_state.class_names, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={
                        "index": "Class",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1-score": "F1-Score",
                        "support": "Support"
                    })

                    styled_df = report_df.style.format({
                        'Precision': '{:.3f}',
                        'Recall': '{:.3f}',
                        'F1-Score': '{:.3f}',
                        'Support': '{:,.0f}'
                    }).set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#3498db'), ('color', 'white'), ('font-weight', 'bold')]},
                        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f9fa')]},
                        {'selector': 'tbody tr:hover', 'props': [('background-color', '#e3f2fd')]},
                    ])

        st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
        

        st.markdown('<div class="section-header">🎯 Average Confusion Matrices</div>', unsafe_allow_html=True)
        
        fig_cols = len(st.session_state.results)
        fig, axs = plt.subplots(1, fig_cols, figsize=(6 * fig_cols, 5))
        if fig_cols == 1:
            axs = [axs]
        
        plot_idx = 0
        for model_key, model_data in st.session_state.results.items():
            avg_cm = np.mean(model_data['conf_matrices'], axis=0)
            if model_key == 'dt':
                title = "Decision Tree"
            elif model_key == 'nb':
                title = "Naive Bayes"
            else:
                title = "CNN + MLP"
            
            plot_confusion_matrix(axs[plot_idx], avg_cm, st.session_state.class_names, title)
            plot_idx += 1
        
        plt.tight_layout()
        st.pyplot(fig)
        

        if 'dt' in st.session_state.results and len(st.session_state.dt_models) > 0:
            st.markdown('<div class="section-header">🌳 Decision Tree Visualization</div>', unsafe_allow_html=True)
            
            all_depths = [model.get_depth() for model in st.session_state.dt_models]
            avg_depth = np.mean(all_depths)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #3498db;">{avg_depth:.1f}</div>
                    <div class="metric-label">Average Depth</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #3498db;">{len(st.session_state.dt_models)}</div>
                    <div class="metric-label">Total Runs</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🌳 Individual Decision Trees")
            
            for run_idx, model in enumerate(st.session_state.dt_models):
                run_number = run_idx + 1
                tree_depth = model.get_depth()
                tree_leaves = model.get_n_leaves()
                run_accuracy = st.session_state.results['dt']['accuracies'][run_idx]
                
                with st.expander(f"🌳 Decision Tree - Run {run_number} (Depth: {tree_depth}, Accuracy: {run_accuracy:.1%})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #3498db;">{tree_depth}</div>
                            <div class="metric-label">Tree Depth</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #3498db;">{tree_leaves}</div>
                            <div class="metric-label">Leaf Nodes</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    try:
                        tree_graph = create_tree_graphviz(model, st.session_state.class_names)
                        st.graphviz_chart(tree_graph.source, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not display decision tree: {str(e)}")
        
        #MORE.....
        st.markdown('<div class="section-header">🧪 Test Additional Images</div>', unsafe_allow_html=True)
        
        additional_test_image = st.file_uploader(
            "Upload a new test image", 
            type=["jpg", "jpeg", "png"], 
            key="additional_test"
        )
        
        if additional_test_image is not None:
            img_rgb, test_features = process_uploaded_image_for_prediction(
                additional_test_image, 
                st.session_state.cnn_model
            )
            
            if img_rgb is not None and test_features is not None:
                predictions = predict_single_image(
                    test_features, 
                    st.session_state.trained_models, 
                    st.session_state.trained_scalers, 
                    st.session_state.class_names
                )
                
                st.markdown("### 🔮 Predictions:")
                
                for model_name, prediction in predictions.items():
                    if model_name == "Decision Tree":
                        icon = "🌳"
                        color = "#3498db"
                    elif model_name == "Naive Bayes":
                        icon = "⚡"
                        color = "#2ecc71"
                    else:
                        icon = "🧠"
                        color = "#e74c3c"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4 style="color: {color}; margin-bottom: 0.5rem;">{icon} {model_name}</h4>
                        <p style="font-size: 1.2rem; font-weight: 600; margin: 0;">Prediction: <span style="color: {color};">{prediction}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🏆 Best Model Summary</div>', unsafe_allow_html=True)
        
        best_acc = max(avg_accuracies)
        best_model_idx = avg_accuracies.index(best_acc)
        best_model_name = model_names[best_model_idx]
        
        if best_model_name == "Decision Tree":
            color = "#3498db"
            icon = "🌳"
        elif best_model_name == "Naive Bayes":
            color = "#2ecc71"
            icon = "⚡"
        else:
            color = "#e74c3c"
            icon = "🧠"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                    padding: 2rem; border-radius: 15px; margin: 2rem 0; 
                    border-left: 5px solid {color}; text-align: center;">
            <h2 style="color: {color}; margin-bottom: 1rem;">{icon} Best Performing Model</h2>
            <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">{best_model_name}</h3>
            <p style="font-size: 1.5rem; font-weight: 600; color: {color};">
                {best_acc*100:.2f}% Average Accuracy
            </p>
            <p style="color: #7f8c8d; margin-top: 1rem;">
                This model achieved the highest average accuracy across all cross-validation runs.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("🔬 Technical Details: How the System Works", expanded=False):
            st.markdown("""
            ## 🧠 System Architecture Overview
            
            ### 📊 **Data Processing Pipeline**
            
            #### 1. **Training Data Preprocessing**
            - **Background Removal**: All training images are preprocessed with AI background removal for optimal performance
            - **Resizing**: All images standardized to 64x64 pixels for consistent processing
            - **Color Space**: Images processed in BGR (OpenCV default) and converted as needed
            - **Normalization**: Pixel values normalized to 0-1 range for neural networks
            - **Quality Assurance**: Clean, consistent dataset with removed backgrounds
            
            #### 2. **User Upload Processing**
            - **AI Background Removal**: Uploaded images automatically processed with rembg AI models
            - **Real-time Processing**: Background removal applied on-the-fly for user uploads
            - **Visual Feedback**: Shows original, mask, and processed images for transparency
            - **Feature Extraction**: Same pipeline as training data for consistency
            
            ### 🤖 **Machine Learning Models**
            
            #### 🌳 **Decision Tree Classifier**
            - **Feature Extraction**: Divides image into 4x4 grid blocks (16 blocks total)
            - **Block Features**: Calculates mean RGB values for each block (48 features total)
            - **Pruning Parameters**:
              - Max Depth: Limits tree complexity to prevent overfitting
              - Min Samples Split: Minimum samples required to split a node
              - Min Samples Leaf: Minimum samples required in leaf nodes
              - Min Impurity Decrease: Minimum improvement required for splits
              - Alpha post pruning: Reduces the complexity and the splitting of same node
            - **Advantages**: Interpretable rules, fast training, visual decision paths
            
            #### ⚡ **Naive Bayes Classifier**  
            - **Feature Extraction**: Statistical features from image pixels
            - **Features Used**:
              - Mean and standard deviation of grayscale values
              - Mean and standard deviation for each RGB channel (6 features total)
            - **Preprocessing**: Features standardized using StandardScaler
            - **Assumptions**: Features assumed independent (naive assumption)
            - **Advantages**: Fast inference, robust to noise, good baseline
            
            #### 🧠 **CNN + MLP (Deep Learning)**
            - **Feature Extractor**: Pre-trained VGG16 CNN (ImageNet weights)
            - **Transfer Learning**: Uses learned features from millions of images
            - **Feature Vector**: Extracts high-dimensional feature representation
            - **MLP Classifier**: Multi-layer perceptron with 128 hidden units
            - **Preprocessing**: VGG16-specific preprocessing (mean subtraction, scaling)
            - **Advantages**: Rich feature representation, high accuracy potential
            
            ### 🔄 **Cross-Validation Process**
            
            #### **Stratified Shuffle Split**
            - **Stratification**: Maintains class distribution in train/test splits
            - **Multiple Runs**: Reduces variance through repeated evaluation
            - **Random Shuffling**: Different data splits for each run
            - **Test Size**: Configurable percentage held out for testing
            - **Identical Splits**: All models use the exact same train/test data for fair comparison
            
            #### **Performance Metrics**
            - **Accuracy**: Percentage of correct predictions
            - **Confusion Matrix**: Shows prediction vs actual class breakdown
            - **Per-Class Metrics**: Precision, recall, F1-score for each fruit type

            ### 🎯 **Background Removal Strategy**
            
            #### **Training Data**
            - **Preprocessed Dataset**: All training images have backgrounds removed beforehand
            - **Consistent Quality**: Ensures uniform data quality across all samples
            - **Optimal Performance**: Models trained on clean, focused fruit images
            - **No Runtime Overhead**: Fast training since no background removal during processing
            
            #### **User Uploads**
            - **Real-time AI Processing**: Uses rembg deep learning models for background removal
            - **Professional Quality**: Same technology used in commercial applications
            - **Automatic Operation**: No manual parameter tuning required
            - **Visual Transparency**: Shows processing steps to user

            ### 🔧 **Implementation Details**
            
            #### **Memory Management**
            - **Caching**: CNN model cached to avoid reloading
            - **Session State**: Results stored in Streamlit session for persistence
            - **Efficient Processing**: Training data processed without background removal overhead
            
            #### **Error Handling**
            - **Image Validation**: Checks for valid image decoding
            - **Fallback Protection**: Graceful handling if AI removal fails
            - **Model Validation**: Ensures models trained before prediction
            
            ### 📈 **Performance Optimization**
            
            #### **Feature Engineering**
            - **Block-based Features**: Spatial information preserved in grid structure
            - **Statistical Features**: Robust descriptors for probabilistic models
            - **Deep Features**: Rich representations from pre-trained networks
            
            #### **Hyperparameter Tuning**
            - **Decision Tree Pruning**: Prevents overfitting on dataset
            - **MLP Architecture**: Single hidden layer balances complexity/performance
            - **Standardization**: Feature scaling for optimal model performance
            
            ### 🎨 **Visualization Features**
            
            #### **Decision Tree Graphs**
            - **Graphviz Rendering**: Interactive tree visualization
            - **Feature Names**: Meaningful labels for tree nodes
            - **Class Emojis**: Visual fruit representations in leaves
            
            #### **Confusion Matrices**
            - **Heatmap Visualization**: Color-coded prediction accuracy
            - **Normalized Values**: Shows prediction probabilities
            - **Per-Model Comparison**: Side-by-side performance analysis
            ---
            """)

def login():

    # Create a centered container for the form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="login-container">
            <div class="login-icon">🔐</div>
            <div class="login-title">Login to Access the Suite</div>
            <div class="login-subtitle">Advanced ML Classification System</div>
            """, unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            login_clicked = st.button("🚀 Login", use_container_width=True)
            
            if login_clicked:
                if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main()
    else:
        login()
