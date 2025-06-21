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
import pandas as pd

# Page Setting up
st.set_page_config(
    page_title="Advanced ML Classification Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS To make webapage look good
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
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
    
    /* Authors Box - Updated to match theme */
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
    
    /* Model Cards */
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
    
    /* Metrics Cards */
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
    
    /* Run Results */
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
    
    /* Decision Tree Container */
    .tree-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin: 2rem 0;
        border: 1px solid #ecf0f1;
    }
    
    /* Progress Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db, #2ecc71);
    }
    
    /* Button Styling */
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
    
    /* Status Messages */
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
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

#-- WHERE TO GET THE DATA FROM AND FROM WHAT FOLDERS TO CHOOSE (YOU CAN CHANGE THIS :))
DATA_DIR = "Data"
CLASSES = ["banana", "blueberries", "pomegranates"]
IMAGE_SIZE = (64, 64)
BLOCKS = (4, 4)

# -- THIS IS USED FOR THE DECISION TREE IN ORDER NOT TO OVERFIT AND MAKE THE TREE HUGE --> LARGE TIME
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
    features = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features.append(np.mean(gray))
    features.append(np.std(gray))
    for i in range(3):
        channel = image[:, :, i]
        features.append(np.mean(channel))
        features.append(np.std(channel))
    return np.array(features)
### DEEP FEATURES FOR MLP 
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
## GET THE DATA BALANCE IT AND SHUFFLE SO WE GET DIFFERENT SPLITS AND CAN GET MORE ACCURATE RESULTS 
def load_image_paths_and_labels():
    X_paths, y, paths = [], [], []
    label_map = {name: idx for idx, name in enumerate(CLASSES)}
    class_files = {}
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_files[cls] = sorted(files)
    min_count = min(len(files) for files in class_files.values())
    
    st.markdown(f"""
    <div class="status-info">
        <strong>📊 Dataset Information:</strong><br>
        Balanced dataset by minimum class size: <strong>{min_count}</strong> images per class.
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
## USED TO PLOT THE DECISON TREE
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

    # Enhanced node styling with better colors
    # Internal nodes - decision nodes
    dot_data = dot_data.replace('fillcolor="#e58139"', 'fillcolor="#3498db", fontcolor="white"')
    dot_data = dot_data.replace('fillcolor="#39e581"', 'fillcolor="#2ecc71", fontcolor="white"') 
    dot_data = dot_data.replace('fillcolor="#8139e5"', 'fillcolor="#9b59b6", fontcolor="white"')
    
    # Leaf nodes - class predictions with fruit-specific colors
    dot_data = dot_data.replace('fillcolor="#e5d439"', 'fillcolor="#f1c40f", fontcolor="black"')  # Banana - yellow
    dot_data = dot_data.replace('fillcolor="#399de5"', 'fillcolor="#3498db", fontcolor="white"')  # Blueberries - blue
    dot_data = dot_data.replace('fillcolor="#e53981"', 'fillcolor="#e74c3c", fontcolor="white"')  # Pomegranates - red

    return graphviz.Source(dot_data)
## THIS IS FOR THE REPORTS LIKE RECALL PRECISISON F1 
def create_classification_report_table(y_true, y_pred, class_names, model_name):
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
## USED TO PLOT THE CONFUSION MATRIX
def plot_confusion_matrix(ax, cm, class_names, title):
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

    
   


## THIS IS THE MAIN WEBPAGE
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Advanced ML Classification Suite</h1>
        <p>Comparative Study of Image Classification Using Decision Tree, Naive Bayes, and CNN+MLP</p>
    </div>
    """, unsafe_allow_html=True)
    
    #START AND SAVE SESSION SO WHEN YOU LOOK AT A TREE OR SOMETHING IT DOESNT RESET
    if 'experiment_completed' not in st.session_state:
        st.session_state.experiment_completed = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'dt_models' not in st.session_state:
        st.session_state.dt_models = []
    if 'last_predictions' not in st.session_state:
        st.session_state.last_predictions = {}
    if 'last_misclassified' not in st.session_state:
        st.session_state.last_misclassified = []
    if 'class_names' not in st.session_state:
        st.session_state.class_names = []
    if 'model_names' not in st.session_state:
        st.session_state.model_names = []
    if 'avg_accuracies' not in st.session_state:
        st.session_state.avg_accuracies = []
    if 'N_RUNS' not in st.session_state:
        st.session_state.N_RUNS = 5
    if 'all_misclassified' not in st.session_state:
        st.session_state.all_misclassified = {}
    
    #------ SIDEBAR ---------------
    st.sidebar.markdown("## ⚙️ Configuration Panel")
    st.sidebar.markdown("---")
    
    #CHOOSE WHAT TO RUN 
    st.sidebar.markdown("### 🤖 Select Models to Run")
    run_decision_tree = st.sidebar.checkbox("🌳 Decision Tree", value=True)
    run_naive_bayes = st.sidebar.checkbox("⚡ Naive Bayes", value=True)
    run_cnn_mlp = st.sidebar.checkbox("🧠 CNN + MLP", value=True)
    
    if not any([run_decision_tree, run_naive_bayes, run_cnn_mlp]):
        st.sidebar.error("⚠️ Please select at least one model to run!")
        return
    
    #PICK NUM OF RUNS
    st.sidebar.markdown("### 🔄 Cross-Validation Settings")
    N_RUNS = st.sidebar.slider("Number of Runs", min_value=1, max_value=10, value=5, step=1)
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    
    ##MORE SETTINGS ALREADY TUNED DONT PLAY WITH THEM UNLESS YOU WANT TO LEARN 
    st.sidebar.markdown("### 🔧 Advanced Settings")
    random_state = st.sidebar.number_input("Random State", min_value=1, max_value=100, value=42)


    #PRUNING OF TREE
    st.sidebar.markdown("#### 🌳 Decision Tree Pruning")
    max_depth = st.sidebar.slider("Max Depth", min_value=2, max_value=20, value=10, step=1,
                        help="Maximum depth of the tree")
    min_samples_split = st.sidebar.slider("Min Samples Split", min_value=0, max_value=50, value=10, step=10, 
                                    help="Minimum samples required to split an internal node")
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=0, max_value=20, value=5, step=5,
                                   help="Minimum samples required to be at a leaf node")
    min_impurity_decrease = st.sidebar.slider("Min Impurity Decrease", min_value=0.01, max_value=0.1, value=0.01, step=0.01,
                                        help="Split only if it decreases impurity by this amount")

    # ]

    #INFORMATION ON THE PRUNING
    st.sidebar.markdown("""
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-size: 0.8rem;">
        <strong>🔍 Pruning for Dataset (249 per class):</strong><br>
        • <strong>Max Depth (10):</strong>to be safe however other factors should make it converge before then<br>
        • <strong>Min Samples Split (10):</strong> Prevents splitting small nodes<br>
        • <strong>Min Samples Leaf (5):</strong> Each leaf must have ≥5 samples<br>
        • <strong>Min Impurity Decrease (0.01):</strong> Split must improve purity by ≥1%<br>
        
        """, unsafe_allow_html=True)
    
    # Check if sidebar parameters changed and reset if needed
    current_params = {
        'run_decision_tree': run_decision_tree,
        'run_naive_bayes': run_naive_bayes, 
        'run_cnn_mlp': run_cnn_mlp,
        'N_RUNS': N_RUNS,
        'test_size': test_size,
        'random_state': random_state,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'min_impurity_decrease': min_impurity_decrease
    }

    if 'previous_params' not in st.session_state:
        st.session_state.previous_params = current_params
    elif st.session_state.previous_params != current_params:
        #IF SOMETHING CHANGED ON SIDEBAR RESET THE SETTINGS
        for key in ['experiment_completed', 'results', 'dt_models', 'last_predictions', 
                   'last_misclassified', 'all_misclassified', 'class_names', 'model_names', 'avg_accuracies', 'N_RUNS']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.previous_params = current_params
        st.rerun()
    
    #AUTHORS BOX
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="authors-box">
        <h3>👥 Project Authors</h3>
        <div class="authors-names">Heba Mustafa & Mohammad Omar</div>
    </div>
    """, unsafe_allow_html=True)
    
    #JUST UI TO MAKE PAGE LOOK NICE 
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
    
    # Start
    st.markdown('<div class="section-header">🚀 Run Experiment</div>', unsafe_allow_html=True)

    # RESET
    col1, col2 = st.columns([3, 1])
    with col1:
        start_experiment = st.button("🎯 Start Classification Experiment", key="start_experiment")
    with col2:
        if st.button("🔄 Reset", key="reset_experiment"):
            # Clear session state
            for key in ['experiment_completed', 'results', 'dt_models', 'last_predictions', 
                       'last_misclassified', 'class_names', 'model_names', 'avg_accuracies', 'N_RUNS']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if start_experiment or st.session_state.experiment_completed:
        if start_experiment:
            #LOAD DATA AND PREPARE FOR RUNNING
            st.session_state.N_RUNS = N_RUNS
            paths, y, label_map = load_image_paths_and_labels()
            st.session_state.class_names = list(label_map.keys())
            
            #PROGRESS BAR
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            #EXTRACT FEATURES
            features_dict = {}
            
            if run_decision_tree:
                status_text.markdown('<div class="status-info">📊 Extracting features for Decision Tree...</div>', unsafe_allow_html=True)
                dt_features = []
                for i, p in enumerate(paths):
                    img = cv2.imread(p)
                    dt_features.append(preprocess_image_blocked_rgb(img))
                    progress_bar.progress((i + 1) / len(paths) * 0.3)
                features_dict['dt'] = np.array(dt_features)
            
            if run_naive_bayes:
                status_text.markdown('<div class="status-info">📊 Extracting features for Naive Bayes...</div>', unsafe_allow_html=True)
                nb_features = []
                for i, p in enumerate(paths):
                    img = cv2.imread(p)
                    nb_features.append(extract_statistical_features(img))
                    progress_bar.progress(0.3 + (i + 1) / len(paths) * 0.3)
                features_dict['nb'] = np.array(nb_features)
            
            if run_cnn_mlp:
                status_text.markdown('<div class="status-info">🧠 Loading CNN feature extractor (VGG16)...</div>', unsafe_allow_html=True)
                cnn_model = get_cnn_feature_extractor()
                
                status_text.markdown('<div class="status-info">🔍 Extracting CNN features...</div>', unsafe_allow_html=True)
                cnn_features = []
                for i, p in enumerate(paths):
                    img = cv2.imread(p)
                    cnn_features.append(extract_cnn_features(cnn_model, img))
                    progress_bar.progress(0.6 + (i + 1) / len(paths) * 0.4)
                
                features_dict['cnn'] = np.array(cnn_features)
            
            #HERE THIS WILL SCALE FEATURES 
            if run_naive_bayes:
                scaler_nb = StandardScaler()
                features_dict['nb_scaled'] = scaler_nb.fit_transform(features_dict['nb'])
        
            if run_cnn_mlp:
                scaler_cnn = StandardScaler()
                features_dict['cnn_scaled'] = scaler_cnn.fit_transform(features_dict['cnn'])
        
        progress_bar.progress(1.0)
        status_text.markdown('<div class="status-success">✅ Feature extraction completed!</div>', unsafe_allow_html=True)
        
        #PUT RESULT IN STORAGE
        st.session_state.results = {}
        st.session_state.dt_models = []  # Store all decision tree models from each run
        if run_decision_tree:
            st.session_state.results['dt'] = {'accuracies': [], 'conf_matrices': []}
        if run_naive_bayes:
            st.session_state.results['nb'] = {'accuracies': [], 'conf_matrices': []}
        if run_cnn_mlp:
            st.session_state.results['mlp'] = {'accuracies': [], 'conf_matrices': []}
        
        st.session_state.last_misclassified = []
        st.session_state.last_predictions = {}
        
        #VALIDATION WILL BE HERE
        st.markdown('<div class="section-header">🔄 Cross-Validation Results</div>', unsafe_allow_html=True)
        
        splitter = StratifiedShuffleSplit(n_splits=N_RUNS, test_size=test_size, random_state=random_state)
        run_i = 1
        
        
        for train_idx, test_idx in splitter.split(list(features_dict.values())[0], y):
            y_train, y_test = y[train_idx], y[test_idx]
            paths_test = [paths[i] for i in test_idx]
            
            run_results = []
            
            #IF YOU VHOOSE DT
            if run_decision_tree:
                X_train_dt, X_test_dt = features_dict['dt'][train_idx], features_dict['dt'][test_idx]
                dt_model = DecisionTreeClassifier(
                    random_state=random_state, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_impurity_decrease=min_impurity_decrease
                )
                dt_model.fit(X_train_dt, y_train)
                st.session_state.dt_models.append(dt_model)  # Store the model
                y_pred_dt = dt_model.predict(X_test_dt)
                acc_dt = accuracy_score(y_test, y_pred_dt)
                st.session_state.results['dt']['accuracies'].append(acc_dt)
                st.session_state.results['dt']['conf_matrices'].append(confusion_matrix(y_test, y_pred_dt))
                run_results.append(("🌳 Decision Tree", acc_dt, "#3498db"))
                if run_i == N_RUNS:
                    st.session_state.last_predictions['dt'] = (y_test, y_pred_dt)
            
            # HERE IS NB
            if run_naive_bayes:
                X_train_nb, X_test_nb = features_dict['nb_scaled'][train_idx], features_dict['nb_scaled'][test_idx]
                nb_model = GaussianNB()
                nb_model.fit(X_train_nb, y_train)
                y_pred_nb = nb_model.predict(X_test_nb)
                acc_nb = accuracy_score(y_test, y_pred_nb)
                st.session_state.results['nb']['accuracies'].append(acc_nb)
                st.session_state.results['nb']['conf_matrices'].append(confusion_matrix(y_test, y_pred_nb))
                run_results.append(("⚡ Naive Bayes", acc_nb, "#2ecc71"))
                if run_i == N_RUNS:
                    st.session_state.last_predictions['nb'] = (y_test, y_pred_nb)
            
            # MLP with CNN features
            if run_cnn_mlp:
                X_train_cnn, X_test_cnn = features_dict['cnn_scaled'][train_idx], features_dict['cnn_scaled'][test_idx]
                mlp_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=random_state)
                mlp_model.fit(X_train_cnn, y_train)
                y_pred_mlp = mlp_model.predict(X_test_cnn)
                acc_mlp = accuracy_score(y_test, y_pred_mlp)
                st.session_state.results['mlp']['accuracies'].append(acc_mlp)
                st.session_state.results['mlp']['conf_matrices'].append(confusion_matrix(y_test, y_pred_mlp))
                run_results.append(("🧠 CNN+MLP", acc_mlp, "#e74c3c"))
                if run_i == N_RUNS:
                    st.session_state.last_predictions['mlp'] = (y_test, y_pred_mlp)
                    
            # Display results for this run in a compact format
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
        
        #FINAL RESULT
        st.session_state.model_names = []
        st.session_state.avg_accuracies = []
        
        for model_key, model_data in st.session_state.results.items():
            avg_acc = np.mean(model_data['accuracies'])
            st.session_state.avg_accuracies.append(avg_acc)
            
            if model_key == 'dt':
                model_name = "Decision Tree"
            elif model_key == 'nb':
                model_name = "Naive Bayes"
            else:
                model_name = "CNN + MLP"
            
            st.session_state.model_names.append(model_name)
        
        #DONE 
        st.session_state.experiment_completed = True
    
    #DISPLAY RESULTS
    if st.session_state.experiment_completed:
        # Results Summary
        st.markdown('<div class="section-header">📈 Final Results Summary</div>', unsafe_allow_html=True)
        
        # Average accuracies
        cols = st.columns(len(st.session_state.results))
        
        for i, (model_key, model_data) in enumerate(st.session_state.results.items()):
            avg_acc = np.mean(model_data['accuracies'])
            
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
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div class="metric-value" style="color: {color};">{avg_acc*100:.2f}%</div>
                    <div class="metric-label">{model_name}</div>
                </div>
                """, unsafe_allow_html=True)
        
        #PUT THE REPORTS AS TABLES 
        st.markdown('<div class="section-header">📄 Classification Reports</div>', unsafe_allow_html=True)

        for model_key, (y_test, y_pred) in st.session_state.last_predictions.items():
            if model_key == 'dt':
                model_name = "🌳 Decision Tree"
            elif model_key == 'nb':
                model_name = "⚡ Naive Bayes"
            else:
                model_name = "🧠 CNN + MLP"
            
            st.markdown(f"#### {model_name}")
            report_df = create_classification_report_table(y_test, y_pred, st.session_state.class_names, model_name)
            
            #STYLING
            styled_df = report_df.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'Support': '{:d}'
            }).set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#3498db'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f9fa')]},
                {'selector': 'tbody tr:hover', 'props': [('background-color', '#e3f2fd')]},
            ])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        #CONFUSION MATRIX 
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
        
        # DT Visualization
        if 'dt' in st.session_state.results and len(st.session_state.dt_models) > 0:
            st.markdown('<div class="section-header">🌳 Decision Tree Visualization</div>', unsafe_allow_html=True)
            
            #ANG DEPTH 
            all_depths = [model.get_depth() for model in st.session_state.dt_models]
            avg_depth = np.mean(all_depths)
            
            #STATS
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
            
            #TREE EXPANDER
            st.markdown("### 🌳 Individual Decision Trees")
            
            for run_idx, model in enumerate(st.session_state.dt_models):
                run_number = run_idx + 1
                tree_depth = model.get_depth()
                tree_leaves = model.get_n_leaves()
                
                #ACC FOR EACH RUN 
                run_accuracy = st.session_state.results['dt']['accuracies'][run_idx]
                
                with st.expander(f"🌳 Decision Tree - Run {run_number} (Depth: {tree_depth}, Accuracy: {run_accuracy:.1%})"):
                    
                    # TREE STATS
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
                    
                    st.markdown("""
                    <div class="tree-container">
                    """, unsafe_allow_html=True)
                    
                    try:
                        #DSLAY TREE
                        tree_graph = create_tree_graphviz(model, st.session_state.class_names)
                        st.graphviz_chart(tree_graph.source, use_container_width=True)
                        
                        # EXPLAN
                        st.markdown(f"""
                        <div style="background: #e3f2fd; padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem; border-left: 4px solid #2196f3;">
                            <h4 style="color: #1976d2; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                                🔍 Decision Tree from Run {run_number}:
                            </h4>
                            <div style="color: #424242; line-height: 1.6;">
                                <p><strong>🔹 Internal Nodes (Blue/Green/Purple):</strong> Show decision rules based on RGB values from image blocks</p>
                                <p><strong>🍎 Leaf Nodes (Colored):</strong> Final predictions with fruit emojis and class probabilities</p>
                                <p><strong>🎨 Node Colors:</strong> Represent the dominant class prediction at each node</p>
                                <p><strong>📊 Values:</strong> Show sample counts and class distributions</p>
                                <p><strong>🌿 Tree Flow:</strong> Follow paths from root to leaves based on feature thresholds</p>
                                <p><strong>📈 Performance:</strong> This tree achieved {run_accuracy:.1%} accuracy (Average: {avg_depth:.1f} depth)</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Could not display decision tree visualization: {str(e)}")
                        st.info("The decision tree was trained successfully, but visualization failed. This might be due to Graphviz installation issues.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        
        
        #BEST MODEL 
        st.markdown('<div class="section-header">🏆 Best Model Summary</div>', unsafe_allow_html=True)
        
        best_acc = max(st.session_state.avg_accuracies)
        best_model_idx = st.session_state.avg_accuracies.index(best_acc)
        best_model_name = st.session_state.model_names[best_model_idx]
        
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
                This model achieved the highest average accuracy across all {st.session_state.N_RUNS} cross-validation runs.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
