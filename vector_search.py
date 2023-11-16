import os
import sys
import torch
import warnings
import pandas as pd
from PIL import Image
from time import time
from docarray import DocList
from utils import color, EmbImg
from vectordb import HNSWVectorDB
from lavis.models import load_model_and_preprocess

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_img(key_img:str):
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_feature_extractor",
                                                         model_type="pretrain", is_eval=True,
                                                         device=device)
    key_img = Image.open(key_img).convert("RGB")
    key_img = vis_processors["eval"](key_img).unsqueeze(0).to(device)
    key_emb = model.extract_features({"image": key_img, "text_input": [""]},mode='image').image_embeds
    key_emb = key_emb.flatten()
    
    return key_emb



def main(folder:str, key_img:str):
    vector_db = HNSWVectorDB[EmbImg](workspace=os.path.join('vector_db',folder))
    encodings_path = os.path.join(folder,'photo_encodings')
    cos = torch.nn.CosineSimilarity(dim=0)

    key_emb = encode_img(key_img) #Encode the key image
    query = EmbImg(text='query',embedding=key_emb)
    results = vector_db.search(inputs=DocList[EmbImg]([query]), limit=250)

    data = [[m.img_name, cos(torch.tensor(m.embedding, device=device),
                             key_emb).detach()]
             for m in results[0].matches]

    #Remove File Extension
    key_name = key_img.split(os.sep)[-1].split('.')[0]

    #Save results to csv
    (pd.DataFrame(data,columns=['img_name','score'])
       .to_csv(os.path.join(encodings_path,f'{key_name}.csv'),
               index=False))

    


if __name__ == '__main__':

    #Ensure directory is passed
    if len(sys.argv) != 3:
        print(f'{color.RED}Usage:{color.YELLOW} python vector_search.py [Folder] [Path to Image]{color.ESC}')
        sys.exit(1)

    folder = os.path.abspath(sys.argv[1])
    img_path  = os.path.abspath(sys.argv[2])

    if not os.path.exists(img_path):
        print(f'{color.RED}{img_path}{color.ESC} does not exist')
        sys.exit(1)
    
    if not os.path.exists(folder):
        print(f'{color.RED}{folder}{color.ESC} does not exist')
        sys.exit(1)

    print(f'{color.CYAN}Starting Process...{color.ESC}')
    s = time()
    main(sys.argv[1],sys.argv[2])
    print(f'{color.GREEN}Finished in {color.YELLOW}{round(time()-s,2)}s{color.ESC}')