import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple,List
from lavis.models import load_model_and_preprocess
from multiprocessing import Process, Manager, cpu_count

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


def split_files(files:list, n_cpu:int=None):
    #Number of processes to launch.
    n = len(files)
    m = min(n_cpu if n_cpu else cpu_count(),n)

    x =[[] for _ in range(m)] 

    #For every message file
    for i in range(n):
        x[i%m].append(files[i])

    return x


def data_worker(proc_num,  return_dict,
                key_img:str,
                encodings_path:str, encodings:List[Tuple[str,str]]):


    #Intialise model
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                      model_type="pretrain", is_eval=True,
                                                                      device=device)

    #Encode the key image-txt pair
    key_img = Image.open(key_img).convert("RGB")
    key_img = vis_processors["eval"](key_img).unsqueeze(0).to(device)

    key_enc = model.extract_features({"image": key_img, "text_input": [""]},mode='image').image_embeds

    cos = torch.nn.CosineSimilarity(dim=1)
    data = torch.zeros((len(encodings)), dtype=torch.float32,device=device)

    for i,(img_name,enc_name) in enumerate(encodings):
        print(f'{str(i).zfill(6)}/{len(encodings)}', end='\r')
      
        #Load tensor
        enc = torch.load(os.path.join(encodings_path,enc_name)).image_embeds

        #Add cosine/similarity score between current and key encoding
        data[i] = cos(key_enc,enc).mean().detach()
        # data[i] = (key_enc - enc)[0].pow(2).sum(dim=1).sqrt().sum().detach()

    result = np.column_stack((encodings, data.detach().cpu()))
    return_dict[proc_num] = result

    return 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(chat_folder,key_img):   
    
    n_cpu = 2 #Optimal number seems to be 2
    key_name = key_img.split(os.sep)[-1].split('.')[0]
    encodings_path = os.path.join(chat_folder,'photo_encodings')
    
    if not os.path.exists(encodings_path):
        print(f'{color.RED}ERROR: `photo_encodings` directory in {chat_folder} is empty!.{color.ESC}')
        return 1

    #Load index of files and split according to thread count
    batch_files = split_files(pd.read_csv(os.path.join(encodings_path,'info.csv')).values.tolist(),
                              n_cpu=n_cpu)

    manager = Manager()
    return_dict = manager.dict()
    data = []

    jobs = [Process(target=data_worker, args=[i,return_dict,
                                              key_img,
                                              encodings_path,files])
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
                                           # vv reverse=True for cosine, False for Euclidian
    data = sorted(data,key=lambda x: x[2], reverse=True)
    df = pd.DataFrame(data,columns=['img_name','img_encoding','score'])
    
    df.to_csv(os.path.join(encodings_path,f'{key_name}.csv'),
              index=False)
    return 0


if __name__ == '__main__':

    #Ensure directory is passed
    if len(sys.argv) != 3:
        print(f'{color.RED}Usage:{color.YELLOW} python brute_search.py [Folder] [Path to Image]{color.ESC}')
        sys.exit(1)

    chat_path = os.path.abspath(sys.argv[1])
    img_path  = os.path.abspath(sys.argv[2])

    if not os.path.exists(img_path):
        print(f'{color.RED}{img_path}{color.ESC} does not exist')
        sys.exit(1)
    
    if not os.path.exists(chat_path):
        print(f'{color.RED}{chat_path}{color.ESC} does not exist')
        sys.exit(1)

    s = time.time()
    main(sys.argv[1],sys.argv[2])
    print(f'{color.GREEN}Finished in {color.YELLOW}{round(time.time()-s,2)}s{color.ESC}')



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