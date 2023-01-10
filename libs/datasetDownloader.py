import gdown
from zipfile import ZipFile
import os
import json


class driveDownloader:
    """
        MSTCD downloader class from google drive 
    """
    def __init__(self, mode, rootDir):
        """
            `mode` : train , validation or test
            'rootDir` : is the path where the `train/validation/test` data store in it

        """
        self.mode = mode
        self.root = rootDir
        with open('libs/config.json') as f:
            self.config = json.load(f)
        if not os.path.isdir(f"{self.root}/Datasets"):
            os.mkdir(f"{self.root}/Datasets")
        self.id = self.config[mode]["id"]
        self.outputPath = f"{self.root}/Datasets/{self.mode}.zip"

        self.cacheDir = f"{self.root}/Datasets/{self.mode}.zip"
        self.cacheIsExist = os.path.exists(self.cacheDir)
        self.fileDir = f"{self.root}/Datasets/{self.mode}"
        self.fileIsExist = os.path.exists(self.fileDir)
        
    def fileDownloader(self):
        try:
            # gdown.download(self.url, self.outputPath, quiet=False,fuzzy=True)
            gdown.download(id=self.id ,output=self.outputPath, quiet=False, fuzzy=True, use_cookies=False)
        except:
            raise Exception('file downloading has been failed')


    def folderDownloader(self):
        try:
            gdown.download_folder(id=self.id, output=self.outputPath, quiet=True, use_cookies=False)
        except:
            raise Exception('folder downloading has been failed')
    
    def unzipFile(self):
        try:
            with ZipFile(self.cacheDir, 'r') as zObject:
                zObject.extractall(path="Datasets/")
            # dirSrc = f"{self.root}/Datasets/" + os.getenv(f'name{self.mode}')
            # dirDst = f"{self.root}/Datasets/" + f'{self.mode}'
            # os.rename(dirSrc, dirDst)
        except:
            raise Exception('extraction has been failed')
