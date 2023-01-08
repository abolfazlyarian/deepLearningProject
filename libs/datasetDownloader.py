import gdown

def fileDownloader(url, output_path):
    gdown.download(url, output_path, quiet=False,fuzzy=True)


def folderDownloader(url, output_path):
    gdown.download_folder(url, output=output_path, quiet=True, use_cookies=False)
