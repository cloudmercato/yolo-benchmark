import os
from urllib.parse import urlparse
import sys
import cv2
import torch
import torchvision
import ultralytics
import yolo_benchmark
import requests
from ultralytics.utils.downloads import is_url


def print_version():
    print(f"version : {yolo_benchmark.__version__}")
    print(f"python_version : {sys.version}".replace('\n', ' '))
    print(f"cv2_version : {cv2.__version__}")
    print(f"torch_version : {torch.version.__version__}")
    print(f"torchvision_version : {torchvision.version.__version__}")
    print(f"ultralytics_version : {ultralytics.__version__}")


DEFAULT_USER_AGENT = "Yolo-Benchmark/1.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)"
DEFAULT_HEADERS = {
    'User-Agent': DEFAULT_USER_AGENT
}


def download(url):
    if is_url(url):
        filename = urlparse(url).path.split('/')[-1]
        dst_path = os.path.join('/tmp/', filename)
        if not os.path.exists(dst_path):
            response = requests.get(url, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            with open(dst_path, 'wb') as fd:
                fd.write(response.content)
        return dst_path
    return url
