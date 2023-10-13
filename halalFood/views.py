from django.shortcuts import render
from django.http import HttpResponse
import pickle
from . import image_processing
import cv2
import numpy as np
from keras.models import load_model
model=load_model('models/model.h5')



def home(request):
    return render(request,'index.html')

def full_prediction(image):
    image_resized=cv2.resize(image,(150,150))
    image_expanded=np.expand_dims(image_resized,axis=0)
    pred=model.predict(image_expanded)
    if pred > 0.5 :
        return "Your image is not clear. Please upload another image." #image not clear
    else:
        halal=image_processing.food_classification(image)
        if halal == True:
            return "The classification of the product is Halel" #image is clear and the food is halal
        else:
            return "The classification of the product is Not Halel" #image is clear and the food is not halal

def byte_image_to_numpy(byte_image):
    np_array = np.frombuffer(byte_image, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

def result(request):
    fileobj=request.FILES['filePath']
    x=fileobj.read()
    x=byte_image_to_numpy(x)
    res=full_prediction(x)
    return render(request,'index.html',{'result' : res})