import os
import numpy as np
import dlib
from sklearn.decomposition import PCA
from skimage import io, color
import matplotlib.pyplot as plt

# dlib의 얼굴 탐지기와 랜드마크 추출기 초기화
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def find_landmarks(img_path):
    # 이미지 로드
    img = io.imread(img_path)
    
    # RGB 이미지를 흑백 이미지로 변환하고 uint8로 변환
    gray_img = (color.rgb2gray(img) * 255).astype(np.uint8)
    
    # 이미지에서 얼굴 탐지
    dets = detector(gray_img, 1)

    # 얼굴이 없는 경우 빈 배열 반환
    if len(dets) == 0:
        return np.empty(0, dtype=int)

    # 랜드마크 좌표를 저장할 배열 초기화
    landmarks = np.zeros((len(dets), 68, 2), dtype=int)
    for k, d in enumerate(dets):
        # 얼굴 영역에서 랜드마크 추출
        shape = sp(gray_img, d)

        # dlib shape를 numpy 배열로 변환하여 저장
        for i in range(0, 68):
            landmarks[k][i] = (shape.part(i).x, shape.part(i).y)

    return landmarks

def encode_faces(img_path, landmarks):
    img = io.imread(img_path)
    face_descriptors = []
    for landmark in landmarks:
        # 얼굴 랜드마크를 dlib.full_object_detection 형태로 변환
        shape = dlib.full_object_detection(
            dlib.rectangle(0, 0, img.shape[1], img.shape[0]),
            [dlib.point(pt[0], pt[1]) for pt in landmark]
        )
        # 얼굴 랜드마크를 사용하여 얼굴의 특징 벡터 계산
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

def calculate_distances(training_data, labels):
    # 얼굴의 특징 벡터 추출
    face_descriptors = []
    for img_path, landmarks in training_data:
        descriptors = encode_faces(img_path, landmarks)
        if len(descriptors) > 0:  # 특징 벡터가 존재하는 경우에만 추가
            face_descriptors.extend(descriptors)
    if len(face_descriptors) == 0:
        raise ValueError("No valid face descriptors found in training data.")

    face_descriptors = np.array(face_descriptors)

    # PCA를 사용하여 특징 벡터 차원 축소
    pca = PCA(n_components=128)
    reduced_face_descriptors = pca.fit_transform(face_descriptors)

    # 대표 벡터 생성
    representative_vector = np.mean(reduced_face_descriptors, axis=0)

    authentic_distances = []
    imposter_distances = []

    # 유클리디안 거리 계산
    for descriptor, label in zip(reduced_face_descriptors, labels):
        distance = np.linalg.norm(descriptor - representative_vector)
        if label == 'JUH': # PSG
            authentic_distances.append(distance)
        else:
            imposter_distances.append(distance)

    return authentic_distances, imposter_distances

def calculate_far_frr(authentic_distances, imposter_distances, threshold):
    far = np.sum(np.array(imposter_distances) < threshold) / len(imposter_distances)
    frr = np.sum(np.array(authentic_distances) > threshold) / len(authentic_distances)
    return far, frr

def find_eer(authentic_distances, imposter_distances):
    min_threshold = min(min(authentic_distances), min(imposter_distances))
    max_threshold = max(max(authentic_distances), max(imposter_distances))
    
    thresholds = np.linspace(min_threshold, max_threshold, 1000)
    far_list = []
    frr_list = []
    
    for threshold in thresholds:
        far, frr = calculate_far_frr(authentic_distances, imposter_distances, threshold)
        far_list.append(far)
        frr_list.append(frr)
    
    # EER을 찾기 위해 FAR와 FRR의 차이가 가장 작은 지점을 찾습니다.
    eer_threshold = thresholds[np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))]
    eer = (far_list[np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))] + 
           frr_list[np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))]) / 2
    
    return eer, eer_threshold, far_list, frr_list, thresholds

# 이미지 로드 및 라벨 생성을 위한 부분
folder = 'linearAlgebra2_face_detection_datasets'
training_data = []
labels = []

for team_folder in os.listdir(folder):
    team_folder_path = os.path.join(folder, team_folder)
    if os.path.isdir(team_folder_path):
        for person_folder in os.listdir(team_folder_path):
            person_folder_path = os.path.join(team_folder_path, person_folder)
            person_initial = person_folder.split("_")[1]  # 개인 이니셜 추출
            if os.path.isdir(person_folder_path):
                for image_name in os.listdir(person_folder_path):
                    image_path = os.path.join(person_folder_path, image_name) # 이미지 경로
                    # 얼굴 랜드마크 찾기
                    landmarks = find_landmarks(image_path)
                    if len(landmarks) == 1:  # 얼굴이 하나라면
                        training_data.append((image_path, landmarks))
                        labels.append(person_initial)
                    else:
                        # 얼굴이 하나가 아닌 경우 경고 출력
                        print(f"얼굴이 하나가 아닌 이미지: {image_path}, 얼굴 개수: {len(landmarks)}")

# 거리 계산
authentic_distances, imposter_distances = calculate_distances(training_data, labels)

# 본인과 타인 분포를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
if authentic_distances:
    plt.hist(authentic_distances, bins=50, density=True, alpha=0.5, color='blue', label='Authentic')
if imposter_distances:
    plt.hist(imposter_distances, bins=50, density=True, alpha=0.5, color='red', label='Imposter')
plt.title('Probability Density Profile')
plt.xlabel('Euclidean Distance')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# EER 계산
eer, eer_threshold, far_list, frr_list, thresholds = find_eer(authentic_distances, imposter_distances)

# EER 시각화
plt.figure(figsize=(10, 6))
plt.plot(thresholds, far_list, label='FAR', color='red')
plt.plot(thresholds, frr_list, label='FRR', color='blue')
plt.axvline(eer_threshold, color='green', linestyle='--', label=f'EER Threshold: {eer_threshold:.4f}')
plt.axhline(eer, color='purple', linestyle='--', label=f'EER: {eer:.4f}')
plt.title('FAR and FRR with EER')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend()
plt.show()

print(f"EER: {eer:.4f} at threshold: {eer_threshold:.4f}")
