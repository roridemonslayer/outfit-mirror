from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch 

processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b2-fashion") #sets up translation
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b2-fashion") #downlaods the trained model 

def segment_image(image_path):
    image = Image.open(image_path) #open image 
    inputs = processor(images =image, return_tensors = "pt")#this process the image so that the model can understan it. 
    with torch.no_grad(): #add the new image to the model 
        outputs = model(**inputs) #this is the output of the mode
    #pick the winning label per pickle
    upsampled= torch.nn.functional.interpolate(
        outputs.logits, #resixing fucntion 
        size = image.size[::-1], #this is the size of the image in reverse order
        mode = "bilinear", #the type of resizing 
        align_corners = False
    )
    predicted_mask= upsampled.argmax(dim = 1)[0] #this is the predicted mask of the image
    outfit_mask = predicted_mask != 0 #this is the mask of the outfit, where 0 is the background and 1 is the outfit.
    return outfit_mask

mask = segment_image("images.jpeg")
print(mask.sum())