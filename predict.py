import argparse
import numpy as np
import json
import torch
from torch import nn
from torchvision import models
from PIL import Image

from collections import OrderedDict



def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

  
    parser.add_argument('--image_path', type=str, default='flowers/test/1/image_06743.jpg', 
                        help='Image to predict')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', 
                        help='trained model')
    parser.add_argument('--top_k', type=int, default=3,
                        help='top classes with high probablities')
    
    parser.add_argument('--category_names', type=str, default= 'cat_to_name.json',
                        help='JSON file containing categories')
   
    parser.add_argument('--gpu', type=int, default=0,
                        help='whether to train on GPU')
    

    # returns parsed argument collection
    return parser.parse_args()
            
def load_model(checkpoint_path):
    
    """
    loads a pretrained model
    Parameters:
     checkpoint - path to model checkpoint
    Returns:
     model - a trained model 
    """
        
        
    chpt = torch.load(checkpoint_path)
    
    if chpt['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.class_to_idx = chpt['class_to_idx']
    
        # classifier
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 512)),
                                    ('relu1', nn.ReLU()),
                                    ('dropout',nn.Dropout(0.2)),
                                    ('fc2', nn.Linear(512, 102)),
                                    ('output',nn.LogSoftmax(dim=1))

                              ]))
    elif chpt['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
           
        model.class_to_idx = chpt['class_to_idx']
        classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(1024, hidden_units)),
                      ('relu', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_units, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))
    else:
        print("Base architecture note recognized")
    
    
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model


def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
   
    
    img = Image.open(image_path)
    
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
   
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
 
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    
    
    img = img.transpose((2, 0, 1))
    
    return img


def predict(image_path, model, category_names, gpu, topk=5):
    """
    Run the prediction
    Parameters:
     image_path - image location to predict
     model - model to use
     category_names = file location cotaining category labels in JSON
     gpu - whether to run on GPU or not
     topk - number of top highest probabilities to predict on
    Returns:
     result - data structure contains the flower name and class probability. and label
    """

    

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    img = process_image(image_path)
    
   
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    if gpu:
        model_input = model_input.to('cuda')
        model.to('cuda')
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    if gpu:
        top_probs = top_probs.detach().numpy().tolist()[0] 
        top_labs = top_labs.detach().numpy().tolist()[0]
    else:
        top_probs = top_probs.detach().cpu().clone().numpy().tolist()[0] 
        top_labs = top_labs.detach().cpu().clone().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    result = []
    c = 0
    for i in range(len(top_probs)):
        result_dict = {}
        result_dict['probability'] = top_probs[i]
        result_dict['labels'] = top_labels[i]
        result_dict['flower_name'] = top_flowers[i]
        i+=1
        result.append(result_dict)
        
#     print(result)
    return result



def main():
    
    
    input_args = get_input_args()
    #load the model
    model = load_model(input_args.checkpoint)
    
    #image to predict
    image_path = input_args.image_path
    if input_args.gpu==1:
        gpu = True
    else:
        gpu = False
    
    print(predict(image_path, model, input_args.category_names, gpu, input_args.top_k))
    
# Call to main function to run the program
if __name__ == "__main__":
    main()