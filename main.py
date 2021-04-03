# frontend/main.py

import requests
import streamlit as st
from PIL import Image
import uuid

import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image

import cv2
import torch
from PIL import Image
from torchvision import transforms

from rounder import rounder
import config

from model import DRModel
import torch
import torch.nn 
import torch.nn.functional as F


# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Diabetic Retinopath prediction web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays the select widget for the styles
#style = st.selectbox("Choose the style", [i for i in STYLES.keys()])

# displays a button
if st.button("Predict on an image"):
    if image is not None :
        
        #res = requests.post(f"http://0.0.0.0:8080/pred", files=files)
        image1 = Image.open(image)
        model = DRModel(arch = config.arch, pre=False)
        #output = inference.inference(model, image)
        model_path= config.MODEL_PATH
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        #height, width = int(image.shape[0]), int(image.shape[1])
        #new_width = int((640 / height) * width)
        #resized_image  = image.resize((224, 224), resample=Image.BILINEAR)
        #resized_image = np.asarray( resized_image, dtype=np.uint8 )
        #resized_image = transforms.ToTensor()(resized_image)
        
        loader = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        image = loader(image1).float()
        #image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  
        #image.cuda()  

        #model.setInput(inp_blob)
        output = model(image)
        output = rounder(output)
        #path = res.json()
        #label = path.get('value')
        #final = path.get(""))
        #st.image(image, width=500)
        st.image(image1)
        st.text(output)
