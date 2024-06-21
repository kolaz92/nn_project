import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from torchvision.models import resnet18, ResNet18_Weights, resnet50
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Предсказание',
)

st.sidebar.success('Выберите нужную страницу')

st.write('# Предсказание класса погоды моделью ResNet18')
st.write('# Предсказание птицы по фото моделью ResNet50')

#Класс модели погоды
class myRegNet(nn.Module):
    def __init__(self):
         super().__init__()
         self.model = resnet18(pretrained=False)
         self.model.fc = nn.Linear(512, 11)
         # замораживаем слои
         for i in self.model.parameters():
             i.requires_grad = False
        # размораживаем только последний, который будем обучать
         self.model.fc.weight.requires_grad = True
         self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
#Класс модели птиц
class myResNet_50(nn.Module):
    def __init__(self):
         super().__init__()
         self.model = resnet50(pretrained=False)
         self.model.fc = nn.Linear(2048, 200)
         # замораживаем слои
         for i in self.model.parameters():
             i.requires_grad = False
        # размораживаем только последний, который будем обучать
         self.model.fc.weight.requires_grad = True
         self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)

def load_and_predict(img,IsWeather=True):
    pred_type = 'weather' if IsWeather else 'bird'

    # Загрузка обученной модели
    Pmodel = myRegNet() if IsWeather else myResNet_50()
    weights = f'model_weights_{pred_type}.pth' if IsWeather else f'model_weights_{pred_type}.pt'

    Pmodel.load_state_dict(torch.load(weights,map_location=torch.device('cpu'))) # модель и веса
    #st.write(type(torch.load(weights,map_location=torch.device('cpu'))))

    Pmodel.eval()

    with open(f'classes_{pred_type}.pkl', 'rb') as file: # словарь классов
        class_to_idx = pkl.load(file)
        class_to_idx = {value:key for key,value in class_to_idx.items()}

    #st.write(class_to_idx)

    class GrayToRGB(object):
        def __call__(self, img):
            if img.mode == 'L':
                img = img.convert('RGB')
            return img
        
    if IsWeather:
        valid_transforms = T.Compose([
            GrayToRGB(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        valid_transforms = T.Compose([  
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(img): # загрузка изображения
        image = valid_transforms(img)        # применение трансформаций
        image = image.unsqueeze(0)      # добавление дополнительной размерности для батча
        return image

    def predict(img):
        img = load_image(img)
        with torch.no_grad():
            output = Pmodel(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = probabilities.argmax().item()
        return predicted_class

    class_prediction = predict(img)
    st.write(f' ### Предсказанный класс: {class_prediction}, Название класса: {class_to_idx[class_prediction]}')

# Функция для первой страницы - Загрузка файла
def upload_img():
    st.title("Загрузка фотографии")

    uploaded_file = st.file_uploader("Загрузите изображение (jpg или png)", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Открываем изображение с помощью PIL
        image = Image.open(uploaded_file)
        st.image(image)
        return image

res = upload_img()

if res is not None:
    if st.button('Предсказать тип погоды'):
        load_and_predict(res,True)

    if st.button('Предсказать птицу'):
       load_and_predict(res,False)