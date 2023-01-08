import gdown
from zipfile import ZipFile
import os
import json


class driveDownloader:
    def __init__(self, mode, rootDir):
        self.mode = mode
        self.root = rootDir
        with open('libs/config.json') as f:
            self.config = json.load(f)
        if not os.path.isdir(f"{self.root}/MSCTDdatasets"):
            os.mkdir(f"{self.root}/MSCTDdatasets")
        self.id = self.config[mode]["id"]
        self.outputPath = f"{self.root}/MSCTDdatasets/{self.mode}.zip"

        self.cacheDir = f"{self.root}/MSCTDdatasets/{self.mode}.zip"
        self.cacheIsExist = os.path.exists(self.cacheDir)
        self.fileDir = f"{self.root}/MSCTDdatasets/{self.mode}"
        self.fileIsExist = os.path.exists(self.fileDir)
        
    def fileDownloader(self):
        try:
            # gdown.download(self.url, self.outputPath, quiet=False,fuzzy=True)
            gdown.download(id=self.id ,output=self.outputPath, quiet=False,fuzzy=True)
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
                zObject.extractall(path="MSCTDdatasets/")
            # dirSrc = f"{self.root}/Datasets/" + os.getenv(f'name{self.mode}')
            # dirDst = f"{self.root}/Datasets/" + f'{self.mode}'
            # os.rename(dirSrc, dirDst)
        except:
            raise Exception('extraction has been failed')
