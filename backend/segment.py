from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch 

processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b2-fashion") #sets up translation
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b2-fashion") #downlaods the trained model 

image = Image.open("images.jpeg") #open image

print(image.size)