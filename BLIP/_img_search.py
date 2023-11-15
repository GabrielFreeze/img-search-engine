import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple,List
from torchvision import transforms
from models.blip import blip_feature_extractor
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


def data_worker(proc_num,  return_dict, key_path:str,
                encodings_path:str,model_url:str,
                encodings:List[Tuple[str,str]]):

    img_sz = 640

    #Intialise model
    model = blip_feature_extractor(pretrained=model_url, image_size=img_sz, vit='base')
    model.eval()
    model = model.to(device)

    #Encode the key image
    key_enc = model(load_img(img_path=key_path,img_sz=img_sz),
                    '',mode='image')[0]


    cos = torch.nn.CosineSimilarity(dim=1)
    data = torch.zeros((len(encodings)), dtype=torch.float32,device=device)
    # data = []

    for i,(img_name,enc_name) in enumerate(encodings):
        print(f'{str(i).zfill(6)}/{len(encodings)}', end='\r')
        #Load tensor
        img_enc = torch.load(os.path.join(encodings_path,enc_name))
        
        #Add cosine score between current and key img
        data[i] = cos(key_enc,img_enc).mean().detach()

    result = np.column_stack((encodings, data.detach().cpu()))
    return_dict[proc_num] = result

    return 0



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():   
    chat_folder = sys.argv[1]
    key_path = sys.argv[2]
    n_cpu = 2 #Optimal number seems to be 2

    key_name = key_path.split(os.sep)[-1].split('.')[0]

    if os.path.exists(os.path.join('checkpoints','model_base_retrieval_coco.pth')):
        model_url = os.path.join('checkpoints','model_base_retrieval_coco.pth')
    else:
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    
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

    jobs = [Process(target=data_worker, args=[i,return_dict,key_path,encodings_path,model_url,files])
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

    # data = []
    # cos = torch.nn.CosineSimilarity(dim=1)

    # #Calculate similarity with every other image in the chat
    # #in order to find the most similar
    # for i,(img_name,enc_name) in enumerate(list(enc_df.itertuples(index=False))):
        
    #     print(f'{str(i).zfill(6)}/{len(enc_df)}', end='\r')
    #     #Load tensor
    #     img_enc = torch.load(os.path.join(encodings_path,enc_name))

    #     #Make sure img_sz is the same when encoding both images
    #     assert img_enc.shape == key_enc.shape

    #     #TODO: Am I calculating the similarity score correctly?
    #     score = cos(key_enc,img_enc).mean().detach().cpu().numpy()

    #     data.append([img_name,enc_name,score])


    data = sorted(data,key=lambda x: x[2], reverse=True)
    df = pd.DataFrame(data,columns=['img_name','img_encoding','score'])
    
    df.to_csv(os.path.join(encodings_path,f'{key_name}.csv'),
              index=False)
    return 0

 

if __name__ == '__main__':
    main()


# feat_1 = model(img_1, '', mode='image')[0]
# feat_2 = model(img_2, '', mode='image')[0]

# cos_0 = torch.nn.CosineSimilarity(dim=0)
# cos_1 = torch.nn.CosineSimilarity(dim=1)

# score_0 = cos_0(feat_1,feat_2).mean()
# score_1 = cos_1(feat_1,feat_2).mean()

# print(score_0,score_1)
# multimodal_feature = model(image, caption, mode='multimodal')[0,0]
# text_feature = model(image, caption, mode='text')[0,0]


#os.path.join('..','checkpoints','blip_retrieval.pth'