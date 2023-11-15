import os
import time
import sys
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
    if len(sys.argv) == 1:
        print(f'{color.RED}ERROR: No chat folder provided.{color.ESC}')
        return 1

        
    full_path = os.path.abspath(sys.argv[1])
    os.chdir(os.path.join(os.getcwd(),'BLIP'))

    s = time.time()
    subprocess.run(f'python _encode_img.py {full_path}')
    print(f'{color.GREEN}Finished in {color.YELLOW}{round(time.time()-s,2)}s{color.ESC}')

    return


if __name__ == "__main__":
    main()
