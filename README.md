This project demonstrates how to generate captions for images using a VisionEncoderDecoderModel and then perform similarity searches using FAISS and LangChain. The images are stored in Google Drive, and the model generates captions that are then embedded and stored in a vector store for efficient similarity searching.

Ensure you have the following installed:

Python 3.7 or higher
PyTorch
Google Colab or equivalent environment for running the code
Access to Google Drive
You can install the necessary libraries using pip
pip install faiss-cpu
pip install langchain
pip install langchain-community
pip install transformers
pip install pillow
pip install matplotlib
pip install sentence-transformers

1. Import Libraries
The following libraries are used in this project:
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from google.colab import drive
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import time


2.Initialize Model, Feature Extractor, and Tokenizer
Initialize the pre-trained VisionEncoderDecoderModel, ViTFeatureExtractor, and AutoTokenizer:
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

3.Set Device
Set the device to GPU if available, otherwise CPU:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

4. Predict Captions
Define a function to predict captions for a list of image paths:
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
5. Mount Google Drive
Mount your Google Drive to access images:
drive.mount('/content/drive')
6. Generate Captions for Images
Generate captions for images stored in a specified directory in Google Drive:
my_drive_path = "/content/drive/MyDrive/pictures"
files = os.listdir(my_drive_path)
images = [file for file in files if file endswith(".jpg") or file.endswith(".png") or file.endswith(".jfif")]

img_caption = {}
for image in images:
    image_path = os.path.join(my_drive_path, image)
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    plt.show()
    caption = predict_step([image_path])[0]
    img_caption[image_path] = caption
    print(caption)
7. Create Documents from Image Paths and Captions
Create documents for each image and its corresponding caption:
documents = [Document(page_content=caption, metadata={"image_path": image_path}) for image_path, caption in img_caption.items()]


