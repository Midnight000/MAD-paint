from .celebahq import load_celebahq, load_lama_celebahq, load_test
from .imagenet import load_imagenet, load_image


REFERENCE_DIRS = {
    "celeba-hq": "datasets/lama-celeba/visual_test_source_256",
    "imagenet": "datasets/imagenet1kval/",
    "imagenet512": "datasets/imagenet1kval/",
}
