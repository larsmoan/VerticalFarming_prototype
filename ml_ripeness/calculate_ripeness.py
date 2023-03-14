
from PIL import Image
import numpy as np
from keras.models import load_model

#Script for processing and doing the prediction using the custom ripeness model - not the ResNet50 model

def process_and_predict(file,model):
    im = Image.open(file)
    width, height = im.size
    if width == height:
        im = im.resize((224,224), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left,top,right,bottom))
            im = im.resize((224,224), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left,top,right,bottom))
            im = im.resize((224,224), Image.ANTIALIAS)
            
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 224, 224, 3)
    
    ripeness = model.predict(ar)
   
    print('Ripeness:', int(ripeness))
    return im.resize((224,224), Image.ANTIALIAS)

if __name__ == '__main__':
    model = load_model('ripeness_model.h5')
    process_and_predict('test.png',model)
