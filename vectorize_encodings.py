import os
import sys
import torch
import warnings
import pandas as pd
from time import time
from docarray import DocList
from utils import color, EmbImg
from vectordb import HNSWVectorDB

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(folder:str):

    vector_db = HNSWVectorDB[EmbImg](workspace=os.path.join('vector_db',folder))

    encodings_path = os.path.join(folder,'photo_encodings')
    emb2img = (pd.read_csv(os.path.join(encodings_path,'info.csv'))
                .set_index('img_encoding')['img']
                .to_dict())

    vector_list = [EmbImg(img_name=emb2img[emb_name],
                        embedding=torch.load(os.path.join(encodings_path,emb_name)).flatten())
                for emb_name in emb2img.keys()]
    vector_db.index(inputs=DocList[EmbImg](vector_list))

    return 0


if __name__ == '__main__':

    #Ensure directory is passed
    if len(sys.argv) == 1:
        print(f'{color.RED}ERROR: No image folder provided.{color.ESC}')
        sys.exit(1)

    s=time()
    main(sys.argv[1])
    print(f'{color.GREEN}Finished in {color.YELLOW}{round(time()-s,2)}s{color.ESC}')

    sys.exit(0)

    



