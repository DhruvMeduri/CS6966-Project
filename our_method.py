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

def assemble_concept(name, id, concepts_path="./data/tcav/image/concepts/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)


model = torchvision.models.googlenet(pretrained=True)
model = model.eval()
'''
#print(model.inception4c)
#for i in model.modules():
#    print(i)

#This is one way to do it 

layer_act = LayerActivation(model, model.inception4c)
#model = model.eval()

zebra_imgs = load_image_tensors('zebra', transform=False)#
zebra_tensors = torch.stack([transform(img) for img in zebra_imgs])
print(zebra_tensors.shape)
#input = torch.randn(2, 3, 32, 32, requires_grad=True)
attribution = layer_act.attribute(zebra_tensors)
print(attribution.shape)
'''


# Now to try and generate the activations with TCAV - This works as well
# Below is the code for our method with the striped concept vs random_0 - This can be extended to other as well - Just figuring it out for this case

concepts_path = "./data/tcav/image/concepts/"

# stripes_concept = assemble_concept("striped", 0, concepts_path=concepts_path)
# zigzagged_concept = assemble_concept("zigzagged", 1, concepts_path=concepts_path)
# dotted_concept = assemble_concept("dotted", 2, concepts_path=concepts_path)

############################# Concepts for the fire engine ########################################
red_concept = assemble_concept("red", 0, concepts_path=concepts_path)
blue_concept = assemble_concept("blue", 1, concepts_path=concepts_path)
yellow_concept = assemble_concept("yellow", 2, concepts_path=concepts_path)
green_concept = assemble_concept("green", 3, concepts_path=concepts_path)


random_0_concept = assemble_concept("random_0", 4, concepts_path=concepts_path)
random_1_concept = assemble_concept("random_1", 5, concepts_path=concepts_path)

layers=['inception4c', 'inception4d', 'inception4e']

mytcav = TCAV(model=model,
              layers=layers,
              layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))

# For getting the activations for the concept

dic = mytcav.generate_activation(layers,stripes_concept)

temp_1 = torch.load('./cav/av/default_model_id/striped-0/inception4d/0.pt')# For striped
temp_2 = torch.load('./cav/av/default_model_id/striped-0/inception4d/1.pt')# For striped
concept_activations = torch.cat((temp_1,temp_2))
concept_activations = np.array(concept_activations)
#print(len(concept_activations))
concept_labels = []
for i in range(len(concept_activations)):
    concept_labels.append(1)
concept_labels = np.array(concept_labels)
#Now for the labels


#For getting the activations of the random set

dic = mytcav.generate_activation(layers,dotted_concept)

temp_1 = torch.load('./cav/av/default_model_id/dotted-2/inception4d/0.pt')# For random concept
temp_2 = torch.load('./cav/av/default_model_id/dotted-2/inception4d/1.pt')# For random concept
random_activations = torch.cat((temp_1,temp_2))
random_activations = np.array(random_activations)
random_labels = []
for i in range(len(random_activations)):
    random_labels.append(-1)
random_labels = np.array(random_labels)

#Now combining everything  for the SVM
activations = np.concatenate((concept_activations,random_activations))
labels = np.concatenate((concept_labels,random_labels))

# Now to the SVM
print("SVM Running")
clf = svm.SVC(kernel='linear',probability=True)
clf.fit(activations, labels)
print(clf.intercept_)

# Now to get the gradients of the zebra class inputs

zebra_imgs = load_image_tensors('zebra', transform=False)#
zebra_tensors = torch.stack([transform(img) for img in zebra_imgs])
#print(zebra_tensors.shape)
#input = torch.randn(2, 3, 32, 32, requires_grad=True)
print("Gradient")
grad= LayerIntegratedGradients(model,model.inception4d)
check = grad.attribute(zebra_tensors,target=340,n_steps=1)
check = torch.reshape(check,(check.shape[0],activations.shape[1]))
check = check.detach().numpy()
print(check.shape)

# This is the notation of the paper
S = clf.decision_function(check)-clf.intercept_
count = 0 
for i in S:
    if i>=0:
        count = count + 1
print("Relative TCAV: ",count/len(S))