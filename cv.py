import cv2
import numpy as np
import os

# Load face images from the dataset
def load_images_and_labels(dataset_path, image_size=(100, 100)):
    face_images = []
    labels = []
    
    for file in os.listdir(dataset_path):
        if file.endswith('.jpg'):
            label = (file.split('_')[1])
            img = cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, image_size)
            face_images.append(img_resized.flatten())
            labels.append(label)
    
    return np.array(face_images), np.array(labels)

dataset_path = r'C:\Users\LeeDongwon\Desktop\workspace\opencv\img'
face_images, labels = load_images_and_labels(dataset_path)

def compute_pca(data, n_components):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    pca_data = np.dot(centered_data, selected_eigenvectors)
    return selected_eigenvectors, mean, pca_data

eigenvectors, mean_face, pca_faces = compute_pca(face_images, n_components=50)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classify(train_data, train_labels, test_data, k=3):
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_data)
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = [distances[i][1] for i in range(k)]
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction

def recognize_faces(eigenvectors, mean_face, train_data, train_labels):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w].flatten()
            face_resized = cv2.resize(face.reshape(h, w), (100, 100)).flatten()
            face_centered = face_resized - mean_face
            face_pca = np.dot(face_centered, eigenvectors)
            
            prediction = knn_classify(train_data, train_labels, face_pca)
            label = f"ID: {prediction}"
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition', img)
        
        if cv2.waitKey(10) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()

# Run face recognition
recognize_faces(eigenvectors, mean_face, pca_faces, labels)
