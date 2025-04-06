
Sameem Fazila <sameemfazila@gmail.com>
Feb 19, 2025, 11:35 AM
to me

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import Counter

# Define paths and categories
dataset_path = r'C:\archive\dataset'
categories = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
img_size = 224
num_folds = 5

# Function to load images
def load_images(dataset_path, categories, img_size):
    images = []
    labels = []
   
    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=(img_size, img_size))
            img = img_to_array(img)
            images.append(img)
            labels.append(category)
   
    return np.array(images), np.array(labels)

# Load dataset
images, labels = load_images(dataset_path, categories, img_size)

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Function for image preprocessing
def preprocess_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    he_img = cv2.equalizeHist(gray_img)
   
    coeffs_h = pywt.wavedec2(gray_img, 'haar', level=1)
    haar_img = pywt.waverec2(coeffs_h, 'haar')
   
    coeffs_d = pywt.wavedec2(gray_img, 'db2', level=2)
    dwt_img = pywt.waverec2(coeffs_d, 'db2')
   
    gaussian_filtered = cv2.GaussianBlur(gray_img, (3, 3), 0.5)
   
    return [gray_img, he_img, haar_img, dwt_img, gaussian_filtered]

# Extract histogram features
def extract_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

# Process images
features = []
for img in images:
    preprocessed = preprocess_image(img)
    features.append(extract_features(preprocessed[0])) # Using grayscale image

features = np.array(features)
features = normalize(features) # Normalize features

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, labels_encoded)

# K-Fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'Boosted Trees': GradientBoostingClassifier()
}

# Training classifiers with k-fold validation
results = {clf: {'accuracy': [], 'sensitivity': [], 'specificity': []} for clf in classifiers}

for train_idx, test_idx in kf.split(X_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
   
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
       
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
       
        sensitivity = np.mean(np.diag(cm) / np.sum(cm, axis=1))
        specificity = np.mean(np.diag(cm) / np.sum(cm, axis=0))
       
        results[name]['accuracy'].append(acc)
        results[name]['sensitivity'].append(sensitivity)
        results[name]['specificity'].append(specificity)

# Compute average metrics
for name in classifiers:
    print(f"{name} - Accuracy: {np.mean(results[name]['accuracy']):.4f}, "
          f"Sensitivity: {np.mean(results[name]['sensitivity']):.4f}, "
          f"Specificity: {np.mean(results[name]['specificity']):.4f}")

# Majority voting classifier
voting_clf = VotingClassifier(estimators=[(name, clf) for name, clf in classifiers.items()], voting='hard')
voting_clf.fit(X_resampled, y_resampled)

# Test image for prediction
test_image_path = r'C:\archive\test\dr.jpg'
test_image = load_img(test_image_path, target_size=(img_size, img_size))
test_image = img_to_array(test_image)
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
test_features = extract_features(test_image_gray).reshape(1, -1)
test_features_normalized = normalize(test_features)

# Predictions
votes = [clf.predict(test_features_normalized)[0] for clf in classifiers.values()]
final_prediction = label_encoder.inverse_transform([Counter(votes).most_common(1)[0][0]])

print(f"Final prediction using majority voting: {final_prediction[0]}")