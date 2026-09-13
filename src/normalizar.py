import numpy as np
from skimage.color import rgb2lab, lab2rgb

def reinhard(X, ref_media, ref_std):
    """Normalizacion de tincion de Reinhard: iguala media y desviacion en LAB.

    Es la mas simple de las normalizaciones de tincion. Macenko y Vahadane
    hacen algo mas sofisticado (separan los vectores de tincion H y E),
    pero Reinhard captura la idea y no necesita deconvolucion.
    """
    Y=np.empty_like(X)
    for k,im in enumerate(X):
        lab=rgb2lab(im.astype(np.float32)/255)
        m,s = lab.mean((0,1)), lab.std((0,1))+1e-6
        lab=(lab-m)/s*ref_std + ref_media
        Y[k]=(np.clip(lab2rgb(lab),0,1)*255).astype(np.uint8)
    return Y

def estadisticas(X):
    L=np.stack([rgb2lab(im.astype(np.float32)/255) for im in X])
    return L.mean((0,1,2)), L.std((0,1,2))


def reinhard_lamina(X, ref_media, ref_std, media_origen=None, std_origen=None):
    """Reinhard con estadisticas de LAMINA, no de parche.

    Diferencia critica: se estima una sola transformacion para toda la
    laminilla y se aplica igual a todos sus parches. Normalizar cada parche
    con sus propias estadisticas borra la senal, porque la densidad nuclear
    -- que es justo lo que distingue tumor de sano -- cambia las
    estadisticas de color del parche.
    """
    if media_origen is None:
        media_origen, std_origen = estadisticas(X)
    Y=np.empty_like(X)
    for k,im in enumerate(X):
        lab=rgb2lab(im.astype(np.float32)/255)
        lab=(lab-media_origen)/(std_origen+1e-6)*ref_std + ref_media
        Y[k]=(np.clip(lab2rgb(lab),0,1)*255).astype(np.uint8)
    return Y
