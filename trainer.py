import os
import numpy as np
import dlib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage import io, color
import joblib

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

def train_classifier(training_data, labels):
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

    # 분류기 훈련
    classifier = SVC(probability=True)
    classifier.fit(reduced_face_descriptors, labels)

    return classifier, pca, representative_vector

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

# 학습기 훈련
classifier, pca, representative_vector = train_classifier(training_data, labels)

# 모델 저장
joblib.dump(classifier, 'classifier.joblib')
joblib.dump(pca, 'pca.joblib')

print("훈련 완료!")
print("대표 벡터:", representative_vector)
