from PIL import Image
import numpy as np 
def get_image_new(image_path,width,height):
    """
    Function to load image from path and rescale it to [-1,1]. 
    """
    image = Image.open(image_path)
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    image = np.multiply(image,2)
    return image