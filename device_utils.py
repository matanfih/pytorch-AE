import torch
import importlib
import time


def get_cpu():
    return "cpu"


def get_gpu(index=0):
    return "cuda:{}".format(index)


def get_freeish_gpu(samples=5, slp=0.3):
    _nvsmi = "nvsmi"
    if importlib.util.find_spec(_nvsmi) is not None:
        import nvsmi

        if not torch.cuda.is_available():
            return None

        gpus = {g.id: g.gpu_util for g in nvsmi.get_gpus()}
        for _ in range(samples - 1):
            time.sleep(slp)
            for t in [(g.id, g.gpu_util) for g in nvsmi.get_gpus()]:
                gpus[t[0]] += t[1] / samples
        return get_gpu(min(gpus, key=gpus.get))

    else:
        print("sadly did not find %s module, will go with default index 0" % _nvsmi)
        return torch.device("cuda:0" if torch.cuda.is_available(_nvsmi) else "cpu")
