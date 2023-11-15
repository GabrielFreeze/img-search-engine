import os
import sys
import time
import torch
import warnings
import pandas as pd
from PIL import Image
from lavis.models import load_model_and_preprocess

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def main(folder:str):
        
    if not os.path.exists(p:=os.path.join(folder,'photo_encodings')):
        os.mkdir(p)

    # txt_df  = pd.read_csv(os.path.join(folder,'txt_info.csv'))
    # img_txt_map = {row[0]:row[1] for _,row in txt_df.iterrows()}
    

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                      model_type="pretrain", is_eval=True,
                                                                      device=device)

    path = os.path.join(folder,'photos')
    img_list = os.listdir(path)
    idx_data:list[tuple[str,str]] = [('','')]*len(img_list)

    for i,img_name in enumerate(img_list):
        
        print(f'{str(i).zfill(6)}/{len(img_list)}', end='\r')


        img = Image.open(os.path.join(path,img_name)).convert("RGB")
        # txt = img_txt_map[img_name]

        img = vis_processors["eval"](img).unsqueeze(0).to(device)
        # txt = txt_processors["eval"](txt)

        feat = model.extract_features({"image": img,"text_input": [
                #txt
                ]},mode='image')

        idx_data[i] = (img_name,f'{i}.pt')
        encodings_path = os.path.join(folder,'photo_encodings')
        torch.save(feat, os.path.join(encodings_path,f'{i}.pt'))

    #Save csv that links images to their encoding
    pd.DataFrame(idx_data, columns=['img','img_encoding']).dropna().to_csv(
        os.path.join(folder,'photo_encodings','info.csv'), index=False)

if __name__ == "__main__":

    #Ensure directory is passed
    if len(sys.argv) == 1:
        print(f'{color.RED}ERROR: No chat folder provided.{color.ESC}')
        sys.exit(1)

    s=time.time()
    main(folder=sys.argv[1])
    print(f'{color.GREEN}Finished in {color.YELLOW}{round(time.time()-s,2)}s{color.ESC}')

    sys.exit(0)
