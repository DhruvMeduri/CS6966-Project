from datasets import load_dataset
import PIL
from PIL import Image  
dataset = load_dataset("HuggingFaceM4/FairFace")
print(dataset['train'][10]['image'])
for i in range(len(dataset['train'])):
    if dataset['train'][i]['race'] == 3:
        dataset['train'][i]['image'].save('./caucasian/' + str(i) + '.jpg')
        #im1 = Image.open(dataset['train'][10]['image'])
#im1.show()