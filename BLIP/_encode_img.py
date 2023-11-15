import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.blip import blip_feature_extractor


class color:
    WHITE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ESC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_img(img_path, img_sz, scl=3):   
    raw_image = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_sz,img_sz),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def main():

    
    folder = sys.argv[1]

    if os.path.exists(os.path.join('checkpoints','model_base_retrieval_coco.pth')):
        model_url = os.path.join('checkpoints','model_base_retrieval_coco.pth')
    else:
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
        
    if not os.path.exists(p:=os.path.join('..',folder,'photo_encodings')):
        os.mkdir(p)

    img_sz = 640
    model = blip_feature_extractor(pretrained=model_url, image_size=img_sz, vit='base')
    model.eval()
    model = model.to(device)

    path = os.path.join('..',folder,'photos')
    img_list = os.listdir(path)
    idx_data:list[tuple[str,str]] = [('','')]*len(img_list)

    for i,img_name in enumerate(img_list):
        
        print(str(i).zfill(5),end='\r')

        img = load_img(img_path=os.path.join(path,img_name),img_sz=img_sz)
        feat = model(img,'',mode='image').detach()

        idx_data[i] = (img_name,f'{i}.pt')
        encodings_path = os.path.join('..',folder,'photo_encodings')

        torch.save(feat, os.path.join(encodings_path,f'{i}.pt'))
        # torch.save(img , os.path.join(encodings_path,f'{i}_img.pt'))

    # #Save any remaining encodings
    # for j in range(batch_sz):
    #     torch.save(feat_data[j], os.path.join(encodings_path,f'{1+i+j-batch_sz}.pt'))
    #     torch.save(img_data[j] , os.path.join(encodings_path,f'{1+i+j-batch_sz}_img.pt'))


    #Save csv that links images to their encoding
    pd.DataFrame(idx_data, columns=['img','img_encoding']).to_csv(
        os.path.join('..',folder,'photo_encodings','info.csv'), index=False)

if __name__ == "__main__":
    main()