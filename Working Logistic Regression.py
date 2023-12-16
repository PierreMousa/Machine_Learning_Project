#       IMPORTS         #
import os
import cv2
import numpy as np
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
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
        extracted_features = extract_features(image)                         #<---    USING FEATURE EXTRACION FUNCTION    #
        
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
X_train, X_test, y_train, y_test = train_test_split(features, labels_numeric, test_size=0.2, random_state=42)



#       MODEL TRAINING  AND TESTING                #
# Initialize and train Logistic Regression model using One-vs-Rest strategy
classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
classifier.fit(X_train, y_train)

# Predictions on the testing set
y_pred = classifier.predict(X_test)




#                REPORT                     #
               # 1 : Number Of Features And Lables/Classes #
num_labels = len(np.unique(labels))
print("number of features : ",max_feature_length)
print(f"Number of Used Labels/Classes: {num_labels}")



                    # 2 : Printed Confusion Matrix(All) #
#Confusion matrix :-
# Decision function for each class
y_scores = classifier.decision_function(X_test)

# Binarize the labels
y_test_binary = label_binarize(y_test, classes=np.unique(labels_numeric))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix(All):
print("Confusion Matrix:")
print(conf_matrix)
                    # 3 : Accuracy #
print(f"Accuracy: {accuracy}")


                    # 4 : Visualized Confusion Matrix(All) #
# Visualize the confusion matrix(All)
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


                    # 5 : Printed And Visualized Confusion Matrix(Each) #
# for each class : 
label_binarizer = LabelBinarizer()
y_test_one_hot = label_binarizer.fit_transform(y_test_binary)
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
# Print the confusion matrix
#print("Confusion Matrix:")
#print(conf_matrix)



                    # 6 : ROC Curve(Each) #
#ROC Curve:-
#ROC Curve for each class
class_names = labels
plt.figure(figsize=(20, 30))
for class_idx in range(y_scores.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_binary[:, class_idx], y_scores[:, class_idx])
    roc_auc = auc(fpr, tpr)
    plt.subplot(2, 3, class_idx + 1)  # Adjust the subplot layout as needed
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
    plt.title(f'ROC Curve - {class_names[class_idx]}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
plt.tight_layout()
plt.show()
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(np.unique(labels_numeric))):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
                    # 7 : ROC Curve(All) #
#ROC Curve for all class
# Plot the ROC curve
plt.figure(figsize=(18, 16))
for i in range(len(np.unique(labels_numeric))):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{np.unique(labels)[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.show()




                    # 8 : loss curve(Each) #
#loss curve:-
                                                     #Note: Based on simple loss curve based on the decision function's output
#Loss Curve for each class :

class_names = np.unique(labels)  
plt.figure(figsize=(20, 30))
for class_idx in range(y_scores.shape[1]):
    plt.subplot(y_scores.shape[1], 1, class_idx + 1)
    plt.plot(np.arange(len(X_test)), np.abs(y_scores[:, class_idx]), label=f'{class_names[class_idx]}')
    plt.title(f'Loss Curve - {class_names[class_idx]}')
    plt.xlabel('Sample Index')
    plt.ylabel('Loss')
    plt.legend()
plt.tight_layout()
plt.show()


                    # 9 : loss curve(All) #
#Loss Curve for all class :
y_scores = np.random.rand(100, 5)
class_names = np.unique(labels)
plt.figure(figsize=(25, 16))
for class_idx in range(y_scores.shape[1]):
    plt.plot(np.arange(len(y_scores)), np.abs(y_scores[:, class_idx]), label=f'{class_names[class_idx]}')
plt.title('Loss Curve for All Classes')
plt.xlabel('Sample Index')
plt.ylabel('Loss')
plt.legend()
plt.show()