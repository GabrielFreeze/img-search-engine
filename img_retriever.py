import os
import sys
import shutil
import pandas as pd
from utils import color

def main(folder, index_file, num_img):
    
    df = pd.read_csv(os.path.join(folder,'photo_encodings',
                                  f'{index_file}.csv'),
                     nrows=int(num_img))

    if not os.path.exists(p:=os.path.join(folder,f'{index_file}_top_{num_img}')):
        os.mkdir(p)

    l = len(df)
    for i,img_name in enumerate(df['img_name']):
        
        print(f"{str(i).zfill(5)}/{l}",end='\r')

        # Copy the file to the new folder
        shutil.copy(os.path.join(folder,'photos',img_name),
                    os.path.join(p,f'{i}_{img_name}.png'))

    return 0


if __name__ == "__main__":

    #Ensure directory is passed
    if len(sys.argv) != 4:
        print(f'{color.RED}Usage: python img_retriever.py {color.YELLOW}[Image Folder] [Name of Index File] [Number of Images]{color.ESC}')
        sys.exit(1)

    folder  = os.path.abspath(sys.argv[1])
    index_file = sys.argv[2]
    num_img    = sys.argv[3]

    if not os.path.exists(folder):
        print(f'{color.RED}{folder} does not exist{color.ESC}')
        sys.exit(1)

                
    if not os.path.exists(p:=os.path.join(folder,'photo_encodings',f'{index_file}.csv')):
        print(f'{color.RED}{p} does not exist{color.ESC}')
        sys.exit(1)


    main(folder, index_file, num_img)
