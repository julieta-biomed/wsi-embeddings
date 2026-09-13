"""Extractores basados en redes preentrenadas.

NO se ejecuta en el notebook: requiere PyTorch y descargar pesos.

Sobre UNI2-h: se distribuye bajo licencia CC-BY-NC-ND 4.0, unicamente para
investigacion academica no comercial y con atribucion. El acceso requiere
registro previo en Hugging Face con correo institucional, y al descargarlo se
acepta no redistribuir los pesos. Este repositorio NO los incluye.
"""
import numpy as np

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESV_IMAGENET  = [0.229, 0.224, 0.225]


def cargar_resnet():
    """ResNet50 preentrenada en ImageNet. Devuelve 2048 dims por parche."""
    import timm
    m = timm.create_model('resnet50', pretrained=True, num_classes=0)
    m.eval()
    return m


def cargar_uni2h(token=None):
    """UNI2-h. Devuelve 1536 dims por parche.

    Requiere acceso concedido en huggingface.co/MahmoodLab/UNI2-h.
    """
    import timm
    from huggingface_hub import login
    if token:
        login(token=token)
    m = timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True,
                          init_values=1e-5, dynamic_img_size=True)
    m.eval()
    return m


def embeddings(parches, modelo, bs=64, media=MEDIA_IMAGENET, desv=DESV_IMAGENET):
    """parches: (N, H, W, 3) uint8. Devuelve (N, D) float32."""
    import torch
    mu = torch.tensor(media).view(1, 3, 1, 1)
    sd = torch.tensor(desv).view(1, 3, 1, 1)
    salida = []
    with torch.no_grad():
        for i in range(0, len(parches), bs):
            x = torch.from_numpy(parches[i:i+bs]).permute(0, 3, 1, 2).float() / 255
            x = (x - mu) / sd
            salida.append(modelo(x).cpu().numpy())
    return np.concatenate(salida)
