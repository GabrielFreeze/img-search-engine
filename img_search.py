import os
import sys
import time
import subprocess

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
    if len(sys.argv) != 3:
        print(f'{color.RED}Usage: python img_search.py {color.YELLOW}[Chat Folder] [Path to Image] {color.ESC}')
        return 1

    chat_path = os.path.abspath(sys.argv[1])
    img_path  = os.path.abspath(sys.argv[2])

    if not os.path.exists(img_path):
        print(f'{color.RED}{img_path}{color.ESC} does not exist')
        return 1
    
    if not os.path.exists(chat_path):
        print(f'{color.RED}{chat_path}{color.ESC} does not exist')
        return 1
              
    
    os.chdir(os.path.join(os.getcwd(),'BLIP'))
    s = time.time()
    subprocess.run(f'python _img_search.py {chat_path} {img_path}')
    print(f'{color.GREEN}Finished in {color.YELLOW}{round(time.time()-s,2)}s{color.ESC}')
    return


if __name__ == "__main__":
    main()
