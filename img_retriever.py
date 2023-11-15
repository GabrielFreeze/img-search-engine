import os
import sys
import shutil
import pandas as pd

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

def main():

    #Ensure directory is passed
    if len(sys.argv) != 4:
        print(f'{color.RED}Usage: python img_retriever.py {color.YELLOW}[Chat Folder] [Name of Index File] [Number of Images]{color.ESC}')
        return 1

    chat_path  = os.path.abspath(sys.argv[1])
    index_file = sys.argv[2]
    num_img    = sys.argv[3]


    if not os.path.exists(chat_path):
        print(f'{color.RED}{chat_path} does not exist{color.ESC}')
        return 1
                
    if not os.path.exists(p:=os.path.join(chat_path,'photo_encodings',f'{index_file}.csv')):
        print(f'{color.RED}{p} does not exist{color.ESC}')
        return 1
    
    df = pd.read_csv(p,nrows=int(num_img))

    if not os.path.exists(p:=f'{index_file}_top_{num_img}'):
        os.mkdir(p)

    l = len(df)
    for i,img_name in enumerate(df['img_name']):
        
        print(f"{str(i).zfill(5)}/{l}",end='\r')

        # Copy the file to the new folder
        shutil.copy(os.path.join(chat_path,'photos',img_name),
                    os.path.join(p,f'{i}.png'))



    return


if __name__ == "__main__":
    main()
