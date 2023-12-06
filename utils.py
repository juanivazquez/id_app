# LIBRERIAS

# generales
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import os
import json
import shutil

# imagenes
import cv2
import PIL
from PIL import Image
from skimage.io import imread, imshow
# from skimage.io import imread as sk_imread
# from skimage.io import imshow as sk_imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate


# modelos
from datasets import load_dataset

# barcode reader
import zxingcpp

# transformers
from transformers import AutoProcessor, Pix2StructForConditionalGeneration




# transformers

repo_id = "juanivazquez/id_card-pix2struct-model-v3"
processor = AutoProcessor.from_pretrained(repo_id)
# processor = AutoProcessor.from_pretrained("google/pix2struct-base")
model = Pix2StructForConditionalGeneration.from_pretrained(repo_id)



from utils_decorators import *



# ------------------------------- MODELO CODIGO DE BARRAS -----------------------------------
@timing_decorator
def BarcodeModel(img):
        results = zxingcpp.read_barcodes(img)
        try:
            output = results[0].text.split('@')
            # print(results[0].text.split('@'),end='\r')
            keys = 'nro_tramite,apellido,nombre,genero,dni,clase,fecha_nac,fecha_emision,num_200'.split(',')
            # if len(output) == len(keys):
            #     output_dict = dict(zip(keys,output))
            # else:
            #     output_dict = {}
            #     for k,v in enumerate(output):
            #         output_dict[k] = v
            output_dict = dict(zip(keys,output))
        except:
            print("Could not find any barcode.",end='\r')  
            output_dict = {}
        
        return output_dict

    # img = cv2.imread(path_name)

# ------------------------------- MODELO TRANSFORMERS  -----------------------------------
def compute(img,np_array=False):
    ''' 
    modelo transformers
    '''
    
    
    
    if np_array==True:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
    
    inputs = processor(images=img, return_tensors="pt")    
    # run Pix2Struct:
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

#@log_execution
def compute2(img):
    ''' 
    compute + ajuste sobre el output: organizar()
    '''
    
    if type(img)==np.ndarray:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
    
    inputs = processor(images=img, return_tensors="pt")    
    # run Pix2Struct:
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    texto = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output = organizar(str([texto]))
    return output


def compute3(img):
    ''' 
    infer from model to dict
    '''
    if not type(img)==np.ndarray:
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = PIL.Image.fromarray(img)
        img = np.asarray(img)
    
    img = rotacion_fina(img)
    
    
    if type(img)==np.ndarray:
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
    inputs = processor(images=img, return_tensors="pt")    
    # run Pix2Struct:
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    texto = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output = organizar(str([texto]))
    return output



# ------------------------------- AJUSTES SOBRE TEXTO  -----------------------------------

def remove_empty_lines(string):
    pattern = r'^\s+(?=\S)'
    return re.sub(pattern, '', string, flags=re.MULTILINE)



def organizar(string):
    '''
    Edita el output del modelo de transformers
    '''
    # string = str([test])# '['+ test + ']'

    # Pattern to match the key-value pairs
    pattern = r"<s_([\w]+)>(.*?)<\/s_([\w]+)>"

    # Find all matches using regular expression
    matches = re.findall(pattern, string)

    # Convert matches to a dictionary with 's_' removed from keys
    dictionary = {key_1: value_1 for key_1, value_1, _ in matches}

    # Check if the last part of the string is missing the closing tag
    if string.endswith("']"):
        # Extract the key and value from the last part of the string
        last_part = string.rsplit("<s_", 1)[1].strip("']")

        # Split the last part into key and value
        last_key, last_value = last_part.split(">", 1)
        last_key = last_key.strip()
        last_value = last_value.strip()

        # Add the key-value pair to the dictionary
        dictionary[last_key] = last_value

    return dictionary


# ------------------------------- AJUSTES SOBRE IMAGEN  -----------------------------------

#deskewing function
def deskew(filename):
    image = imread(filename, as_gray=True)

    #threshold to get rid of extraneous noise
    thresh = threshold_otsu(image)
    normalize = image > thresh

    # gaussian blur
    blur = gaussian(normalize, 3)

    # canny edges in scikit-image
    edges = canny(blur)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1)/(x2 - x1) if (x2-x1) else 0 for (x1,y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)
    
    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90-rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return rotation_number


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
    
 
    
    
# UBICACION DE MODELO
deploy = r'C:\Users\juani\Documents\3_My_Jupiter_Notebooks\5_Galicia\4_lector_de_dnis\caffe_model\deploy.prototxt'
caffemodel = r'C:\Users\juani\Documents\3_My_Jupiter_Notebooks\5_Galicia\4_lector_de_dnis\caffe_model\res10_300x300_ssd_iter_140000.caffemodel'


def best_detection_confidence(image):
    # Model architecture
    prototxt = r'C:\Users\juani\Documents\3_My_Jupiter_Notebooks\5_Galicia\4_lector_de_dnis\caffe_model\deploy.prototxt'
    # Weights
    model = r'C:\Users\juani\Documents\3_My_Jupiter_Notebooks\5_Galicia\4_lector_de_dnis\caffe_model\res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    image_resized = cv2.resize(image, (300, 300))
    # Create a blob
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104, 117, 123))
    # print("blob.shape: ", blob.shape)
    blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])
    # ------- DETECTIONS AND PREDICTIONS ----------
    net.setInput(blob)
    detections = net.forward()
    best = detections[0][0].copy()
    best_detection_confidence = best[best[:,2].argsort()][-1][2]
    
    # for detection in detections[0][0]:
    #      if detection[2] > 0.5:
    #         print(detection[2])
    #         box = detection[3:7] * [width, height, width, height]
    return best_detection_confidence

# img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# img_180 = cv2.rotate(image, cv2.ROTATE_180)
# img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def search_orientation(img):
    '''
    busca la orientación que mejor reconozca un rostro humano mediante rotaciones de 90 grados
    '''
    # plt.imshow(img)
    res_ls = []
    best = best_detection_confidence(img)
    res_ls.append([best,img])
    for i in range(3):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        best = best_detection_confidence(img)
        res_ls.append([best,img])
    res_ls = sorted(res_ls, key=lambda x: x[0])
    print(f'max: {int(res_ls[3][0]*100)}',end='\r')
    output = res_ls[3][1]
    plt.imshow(output)
    return output

def get_face(image,up=0.25,side=0.25):
    
    
    # Model architecture
    prototxt = r'C:\Users\juani\Documents\3_My_Jupiter_Notebooks\5_Galicia\4_lector_de_dnis\caffe_model\deploy.prototxt'
    # Weights
    model = r'C:\Users\juani\Documents\3_My_Jupiter_Notebooks\5_Galicia\4_lector_de_dnis\caffe_model\res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    image_resized = cv2.resize(image, (300, 300))
    # Create a blob
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104, 117, 123))
    blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])
    # ------- DETECTIONS AND PREDICTIONS ----------
    net.setInput(blob)
    detections = net.forward()
    best = detections[0][0].copy()
 
    # ------- NOS QUEDAMOS CON LA MEJOR ----------
    height, width =  image.shape[0:2]
    best_location = best[best[:,2].argsort()][-1][3:7]
    box = best_location * [width, height, width, height]
    box = [ int(x) for x in box ]
    crop_img = image[box[1]-int((box[3]-box[1])*up):box[1]+int((box[3]-box[1])*(1+up)), box[0]-int((box[2]-box[0])*side):box[0]+int((box[2]-box[0])*(1+side))]
    return crop_img



def rotacion_fina(img):
    # image = imread(filename, as_gray=True)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #threshold to get rid of extraneous noise
    thresh = threshold_otsu(image)
    normalize = image > thresh

    # gaussian blur
    blur = gaussian(normalize, 3)

    # canny edges in scikit-image
    edges = canny(blur)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1)/(x2 - x1) if (x2-x1) else 0 for (x1,y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)
    
    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90-rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    # return rotation_number
    result = rotate_image(img,rotation_number)
    return result

