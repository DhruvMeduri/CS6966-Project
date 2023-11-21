import numpy as np
import os, glob
import random
import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ttest_ind

# ..........torch imports............
import torch
import torchvision

from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients, LayerActivation

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

# SVM import

from sklearn import svm

# Method to normalize an image to Imagenet mean and standard deviation
def transform(img):

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


# Now let's define a few helper functions.

# In[3]:


def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def load_image_tensors(class_name, root_path='./data/tcav/image/imagenet/', transform=True):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.jpg')

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(transform(img) if transform else img)
    
    #print(tensors)
    
    return tensors

def assemble_concept(name, id, concepts_path="./data/tcav/image/imagenet/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)


model = torchvision.models.googlenet(pretrained=True)
model = model.eval()

#print(model.inception4c)
#for i in model.modules():
#    print(i)

#This is one way to do it 

layer_act = LayerActivation(model, model.inception4c)
#model = model.eval()

zebra_imgs = load_image_tensors('zebra', transform=False)#
zebra_tensors = torch.stack([transform(img) for img in zebra_imgs])
#print(zebra_tensors.shape)
#input = torch.randn(2, 3, 32, 32, requires_grad=True)
attribution = layer_act.attribute(zebra_tensors)
attribution = torch.reshape(attribution,(attribution.shape[0],attribution.shape[1]*attribution.shape[2]*attribution.shape[3]))
print(attribution[55])


#The other method

concepts_path = "./data/tcav/image/imagenet/"

stripes_concept = assemble_concept("zebra", 0, concepts_path=concepts_path)


layers=['inception4c', 'inception4d', 'inception4e']

mytcav = TCAV(model=model,
              layers=layers,
              layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))

# For getting the activations for the concept

dic = mytcav.generate_activation(layers,stripes_concept)
temp_1 = torch.load('./cav/av/default_model_id/zebra-0/inception4c/0.pt')# For checking
print(temp_1[55])