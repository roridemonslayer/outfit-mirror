from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch 

processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b2-fashion") #sets up translation
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b2-fashion") #downlaods the trained model 

# garment groups, using the model's label ids (see model.config.id2label)
UPPER = {1, 2, 3, 4, 5, 6, 10, 13}  # shirt, top, sweater, cardigan, jacket, vest, coat, cape
LOWER = {7, 8, 9}                  # pants, shorts, skirt
ONE_PIECE = {11, 12}               # dress, jumpsuit
BELT, SHOE = 20, 24
NOT_OUTFIT = {0, 14, 19, 27}       # background/skin, glasses, watch, umbrella


def segment_labels(image):
    """Label every pixel of a PIL image with a garment id. Returns a (height, width) numpy array."""
    return _logits_to_labels(image).numpy()


def segment_image(image_path):
    image = Image.open(image_path).convert("RGB") #open image 
    return _logits_to_labels(image) != 0 #this is the mask of the outfit, where 0 is the background and 1 is the outfit.


def _logits_to_labels(image):
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
    return predicted_mask

if __name__ == "__main__":
    mask = segment_image("images.jpeg")
    print(mask.sum())