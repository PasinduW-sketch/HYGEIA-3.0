import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Setup visual styling
plt.style.use('ggplot')
sns.set_palette('viridis')

def extract_features(df, base_columns):
    """
    Extracts advanced clinical features: Trends (diff), Mean (rolling), and Variance (rolling).
    """
    df_features = df.copy()
    for feat in base_columns:
        # Calculate moving averages (mean) to capture clinical stability
        df_features[f"{feat}_mean"] = df_features[feat].rolling(window=3, min_periods=1).mean()
        # Calculate volatility (variance) to detect unstable vitals
        df_features[f"{feat}_variance"] = df_features[feat].rolling(window=3, min_periods=1).var().fillna(0)
        # Calculate direction of change (trend)
        df_features[f"{feat}_trend"] = df_features[feat].diff().fillna(0)
    return df_features

def get_clinical_label(row):
    """
    Maps hospital-grade (Original) vitals to the 4 requested clinical classes:
    Normal, Fever, Hypoxia, Abnormal/ Critical.
    """
    t, s, h = row['Orig_Temp'], row['Orig_SpO2'], row['Orig_HR']
    if s < 90 or t > 39 or h > 120 or h < 50:
        return 'Abnormal/ Critical'
    elif s < 92:
        return 'Hypoxia'
    elif t >= 38:
        return 'Fever'
    else:
        return 'Normal'

def main():
    print("=== 1. Data Acquisition & Sensor Merging ===")
    df_raw = pd.read_csv('readings.csv', skiprows=2, header=None)
    
    # Original Medical Sensor (Ground Truth)
    original_sensor = df_raw.iloc[:, [0, 1, 2]].apply(pd.to_numeric, errors='coerce')
    original_sensor.columns = ['Orig_Temp', 'Orig_SpO2', 'Orig_HR']
    
    # Hygeia Glove Sensor (Our testing target)
    glove_sensor = df_raw.iloc[:, [3, 4, 5]].apply(pd.to_numeric, errors='coerce')
    glove_sensor.columns = ['Glove_Temp', 'Glove_SpO2', 'Glove_HR']
    
    df_comb = pd.concat([original_sensor, glove_sensor], axis=1).dropna().reset_index(drop=True)
    print(f"Loaded {len(df_comb)} valid dual-sensor readings.")

    print("\n=== 2. Clinical Labeling (Target Generation) ===")
    # Creating the absolute truth labels using the original sensor
    df_comb['Status'] = df_comb.apply(get_clinical_label, axis=1)
    print(df_comb['Status'].value_counts())

    print("\n=== 3. Advanced Feature Extraction ===")
    # Extract Trends, Mean, and Variance strictly for the Glove input
    glove_base_cols = ['Glove_Temp', 'Glove_SpO2', 'Glove_HR']
    df_enhanced = extract_features(df_comb, glove_base_cols)
    
    # Our final training data consists only of the extracted Glove features
    glove_features = [col for col in df_enhanced.columns if 'Glove' in col]
    X_raw = df_enhanced[glove_features]
    y_raw = df_enhanced['Status']

    print(f"Total features utilized per sample: {len(glove_features)}")

    print("\n=== 4. Preprocessing & Augmentation ===")
    # Augment the data slightly by 20x to give LSTM and RF enough variance to study properly
    augmented_features = []
    augmented_labels = []
    for _ in range(20):
        noise = np.random.normal(0, 0.05, X_raw.shape)
        augmented_features.append(X_raw.values + noise)
        augmented_labels.append(y_raw.values)
        
    X_massive = np.vstack(augmented_features)
    y_massive = np.concatenate(augmented_labels)
    
    # Encoding and Scaling
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_massive)
    
    X_train, X_test, y_train, y_test = train_test_split(X_massive, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Ready for training with {len(X_train_scaled)} synthetic patient scenarios.")

    print("\n=== 5. Multi-Model Training (LR, DT, RF, LSTM) ===")
    results = {}
    trained_models = {}

    # 5a. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)
    trained_models['Logistic Regression'] = lr

    # 5b. Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_pred_dt = dt.predict(X_test_scaled)
    results['Decision Tree'] = accuracy_score(y_test, y_pred_dt)
    trained_models['Decision Tree'] = dt

    # 5c. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = accuracy_score(y_test, y_pred_rf)
    trained_models['Random Forest'] = rf

    # 5d. LSTM (Deep Learning)
    print("Training LSTM (Deep Learning)...")
    # LSTM requires 3D formulation: [Samples, Timesteps, Features]
    X_train_lstm = np.expand_dims(X_train_scaled, axis=1)
    X_test_lstm = np.expand_dims(X_test_scaled, axis=1)

    lstm = Sequential([
        LSTM(64, input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Training LSTM over 50 Epochs
    history = lstm.fit(X_train_lstm, y_train, epochs=50, validation_data=(X_test_lstm, y_test), verbose=0)
    
    y_pred_lstm_probs = lstm.predict(X_test_lstm, verbose=0)
    y_pred_lstm = np.argmax(y_pred_lstm_probs, axis=1)
    results['LSTM'] = accuracy_score(y_test, y_pred_lstm)
    trained_models['LSTM'] = lstm

    print("\n=== 6. Clinical Performance Analytics ===")
    for model_name, acc in results.items():
        print(f"{model_name} Accuracy: {acc*100:.2f}%")

    # Final Audit with the Best Model
    best_model_name = max(results, key=results.get)
    print(f"\n=> Top Performing Model: {best_model_name}")

    if best_model_name == 'LSTM':
        best_preds = y_pred_lstm
    else:
        best_preds = trained_models[best_model_name].predict(X_test_scaled)

    print("\n--- CLASSIFICATION REPORT FOR GLOVE PREDICTIONS ---")
    print(classification_report(y_test, best_preds, target_names=le.classes_))

    # Visualization 1: Compare the Models
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='magma')
    plt.title('Accuracy Validation Specifically for Hygeia Glove')
    plt.ylabel('Accuracy Score')
    plt.ylim(0, 1.1)

    # Visualization 2: Confusion Matrix of the Best Model
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.ylabel('Clinical Ground Truth (Original)')
    plt.xlabel('Hygeia Glove Predictions')
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png')
    print("Visualizations saved as 'model_comparison_results.png'")
    plt.show()

if __name__ == '__main__':
    main()
