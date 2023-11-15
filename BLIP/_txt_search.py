import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
from models.blip_itm import blip_itm
from torch.nn.functional import softmax
from multiprocessing import Process, Manager, cpu_count
from torchvision.transforms.functional import InterpolationMode

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
def split_files(files:list, n_cpu:int=None):
    #Number of processes to launch.
    n = len(files)
    m = min(n_cpu if n_cpu else cpu_count(),n)

    x =[[] for _ in range(m)] 

    #For every message file
    for i in range(n):
        x[i%m].append(files[i])

    return x
def data_worker(proc_num,  return_dict, key:str,
                encodings_path:str,model_url:str,
                encodings:List[Tuple[str,str]]):

    img_sz = 640

    #Intialise model
    model = blip_itm(pretrained=model_url,image_size=img_sz, vit='base')
    model.eval()
    model = model.to(device)

    data = torch.zeros((len(encodings)), dtype=torch.float64,device=device)

    for i,(img_name,enc_name) in enumerate(encodings):
        print(f'{str(i).zfill(6)}/{len(encodings)}', end='\r')
        
        #Load tensor
        img_enc = torch.load(os.path.join(encodings_path,f'{enc_name[:-3]}_img.pt'))

        #Calculate img-txt similarity
        score = model(img_enc,key,match_head='itm')
        data[i] = softmax(score,dim=1)[0,1].detach()

    result = np.column_stack((encodings, data.detach().cpu()))
    return_dict[proc_num] = result

    return 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():   
    chat_folder = sys.argv[1]
    key = sys.argv[2]
    n_cpu = 1
    
    #Get model checkpoint
    if os.path.exists(os.path.join('checkpoints','model_base_retrieval_coco.pth')):
        model_url = os.path.join('checkpoints','model_base_retrieval_coco.pth')
    else:
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    
    #Get photo encodings
    encodings_path = os.path.join('..',chat_folder,'photo_encodings')
    if not os.path.exists(encodings_path):
        print(f'{color.RED}ERROR: `photo_encodings` directory in {chat_folder} is empty!.{color.ESC}')
        return 1

    #Load index of files and split according to thread count
    batch_files = split_files(pd.read_csv(os.path.join(encodings_path,'info.csv')).values.tolist(),
                              n_cpu=n_cpu)

    manager = Manager()
    return_dict = manager.dict()
    data = []

    jobs = [Process(target=data_worker, args=[i,return_dict,key,encodings_path,model_url,files])
            for i,files in enumerate(batch_files)]

    #Launch worker for every core
    for p in jobs:
        p.start()

    #Wait for all workers to finish
    for p in jobs:
        p.join()


    #Collate output of workers
    for d in return_dict.values():
        data.extend(d)

    data = sorted(data,key=lambda x: x[2], reverse=True)
    df = pd.DataFrame(data,columns=['img_name','img_encoding','score'])
    
    df.to_csv(os.path.join(encodings_path,f'{key}.csv'),
              index=False)
                      
    return 0


if __name__ == '__main__':
    main()




