from docarray import BaseDoc
from docarray.typing import NdArray



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


class EmbImg(BaseDoc):
    img_name: str = ""
    embedding: NdArray[32*768] #<-- Needs to match shape of stored embeddings