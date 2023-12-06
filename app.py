# 3
import streamlit as st
import pandas as pd
from PIL import Image

from utils import *
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from datasets import load_dataset, load_from_disk

import os
import pickle
import io
import base64
# get icons at: https://icons8.com/
    
#####
#######################################################################################
st.set_page_config(layout="wide")

banner_image = Image.open("custom/banner.jpeg")  
st.image(banner_image, use_column_width=True)

# st.header("DNI READER")    
# st.title("Lector DNI")
st.subheader("Modelo Pix-2-Struct")
st.write("In the realm of language and images, Pix2Struct stands out as a game-changer. Picture this: a model that not only comprehends text but seamlessly decodes visuals like images, tables, and buttons on a webpage. Its unique training involved mastering simplified web page structures, and now, it's your go-to expert for diverse tasks. From interpreting documents to navigating user interfaces, Pix2Struct's versatility shines. It's like having a skilled teammate, excelling effortlessly across different challenges in the language and visual domain")

url = 'https://arxiv.org/abs/2210.03347'
st.write("[paper](%s)" % url)
#######################################################################################    

# Sidebar content
logo_image = Image.open("custom/space_cat.jpg")#"custom/cat_3_circle.png")  
# st.image(logo_image, caption="Your Logo", use_column_width=False, width=150)
st.sidebar.image(logo_image, use_column_width=False, width=200) # caption="JIVB", 


# st.sidebar.header("Sidebar Title")
# st.sidebar.subheader("Subheading")
# st.sidebar.text("Sidebar content goes here.")
st.sidebar.info('Juani Vazquez Broquá')

cols1, cols2 = st.sidebar.columns(2)
cols1.markdown("[![Foo](https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-48.png)](https://www.linkedin.com/in/jivb/)")
cols2.markdown("[![Foo](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/juanivazquez)")


# url = 'https://www.linkedin.com/in/jivb/'
# st.sidebar.text("JIVB")
# st.write("[paper](%s)" % url)




### CARGA DE MODELO
processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
model = Pix2StructForConditionalGeneration.from_pretrained("juanivazquez/id_card-pix2struct-model-v2")




path = os.getcwd()

# path+'\\
# cargamos el modelo

@st.cache_data()
def load_model():
    processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
    model = Pix2StructForConditionalGeneration.from_pretrained("juanivazquez/id_card-pix2struct-model-v2")
    return processor, model

processor, model = load_model()



uploaded_file = st.file_uploader("Upload an image in jpg format", type=["jpg"]) 



# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=['document','file_name', 'apellido', 'nombre','dni','fecha_nac','fecha_emision'])

# Button to download DataFrame as CSV
if st.button("Download Results as CSV"):
    csv_data = st.session_state.results.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # inputs = processor(images=img, return_tensors="pt")
    # generated_ids = model.generate(**inputs, max_new_tokens=50)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    file_dict = compute3(img)
    file_dict['document'] = len(st.session_state.results) + 1
    file_dict['file_name'] = uploaded_file.name
    new_results = pd.DataFrame([file_dict])
    st.image(img, caption="Uploaded Image", use_column_width=True, width=300)
    # st.image(get_face(img))
    st.write('Pix-2-Struct')
    # st.write(f"{file_dict} ")
    # Update the results DataFrame
    # Concatenate the new result to the existing results DataFrame
    st.session_state.results = pd.concat([st.session_state.results, new_results], ignore_index=True)

    # Display the updated results table
    st.table(st.session_state.results.reset_index(drop=True))  # Reset index for display
    # face = np.array(Image.open(uploaded_file))
    # from matplotlib import image
    # from matplotlib import pyplot
    # import cv2
    # # load image as pixel array
    # # face = image.imread(uploaded_file)
    # face = cv2.imread(uploaded_file, 0) 
    # face = get_face(face)
    # face = PIL.Image.fromarray(face)
    # st.image(get_face(face))
    
    
