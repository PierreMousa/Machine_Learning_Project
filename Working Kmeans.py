#       IMPORTS         #
import os
import cv2
import numpy as np
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, adjusted_rand_score
from sklearn.preprocessing import LabelBinarizer

#       FEATURE EXTRACTION FUNCTION         #
# Function to extract HOG features from an image
def extract_features(image):
    hog_features, _ = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return hog_features

#        DATASET LOADING AND READING         #

# Loading the dataset
dataset_path = 'C:\programing\Ml project\Dataset3'
class_folders = os.listdir(dataset_path)

# Lists to store features and labels
features = []
labels = []

# Loop through each class folder
for class_folder in class_folders:
    class_path = os.path.join(dataset_path, class_folder)
    
    # Loop through each image in the class folder
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        
        # Extract HOG features
        extracted_features = extract_features(image)                       #<---    USING FEATURE EXTRACION FUNCTION    #
        
        # Append features to the list
        features.append(extracted_features)
        
        # Append the label to the list
        labels.append(class_folder)


#              LIST PREPROCESSING               #
# This Feature List is a list of lists where each inner list is a sequence of varying length
# Flatten the nested structures in features
flattened_features = [np.array(feature).flatten() for feature in features]

# Find the maximum length of features
max_feature_length = max(len(feature) for feature in flattened_features)

# Pad sequences to the maximum length
padded_features = pad_sequences(flattened_features, maxlen=max_feature_length, padding='post', truncating='post', dtype='float32')

# Convert to NumPy array
features = np.array(padded_features)

"""
Check the shape of features
print("Shape of features after padding:", features.shape)

"""
labels = np.array(labels)



#                USING ENCODERS             #
# Use LabelEncoder to convert class names into numeric labels
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(labels)



#                DATA SPLETING                  #
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_numeric, test_size=0.2, random_state=0)



#       MODEL TRAINING  AND TESTING                #
# Initialize and train KMeans model
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_train)

# Predict clusters on the testing set
y_pred = kmeans.predict(X_test)



#                REPORT                     #
               # 1 : Number Of Features And Lables/Classes #
# Evaluate the clustering performance
ari = adjusted_rand_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the number of labels and extracted features
num_labels = len(np.unique(labels))
num_features = X_train.shape[1]
print(f"Number of Used Labels/Classes: {num_labels}")
print(f"Number of Extracted Features: {num_features}")

                    # 2 : Printed Confusion Matrix(All) #
#Confusion matrix :-

# Print the confusion matrix(All)
print("Confusion Matrix:")
print(conf_matrix)

                    # 3 : Accuracy #
# Print the ARI
print(f"Adjusted Rand Index: {ari}")

                    # 2 : Visualized Confusion Matrix(All) #
# Visualize the confusion matrix (All Classes together)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = label_encoder.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


                    # 2 : Printed And Visualized Confusion Matrix(Each) #
# for each class : 
class_names = np.unique(labels) 
label_binarizer = LabelBinarizer()
y_test_one_hot = label_binarizer.fit_transform(y_test)
y_pred_one_hot = label_binarizer.transform(y_pred)

lass_names = label_binarizer.classes_

plt.figure(figsize=(8, 8))

for class_idx in range(len(class_names)):
    cm = confusion_matrix(y_test_one_hot[:, class_idx], y_pred_one_hot[:, class_idx])
    
    plt.subplot(2, 3, class_idx + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title(f'Confusion Matrix - {class_names[class_idx]}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    print(f'Confusion Matrix - {class_names[class_idx]}:')
    print(cm)
    print('\n')  

plt.tight_layout()
plt.show()
#%%capture