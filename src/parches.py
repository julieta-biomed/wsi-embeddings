"""Genera parches etiquetados de tejido sano y tumoral, con variacion de tincion."""
import numpy as np
from sintetica import tejido_he

def lote(n, densidad, semilla, P=96, tincion=None):
    """n parches de PxP con la densidad nuclear dada.

    tincion: (escala_R, escala_G, escala_B) para simular otro laboratorio.
    """
    rng=np.random.default_rng(semilla)
    X=np.zeros((n,P,P,3),np.uint8)
    grande,_ = tejido_he(P*8, P*8, semilla=semilla, densidad_nuclear=densidad)
    for k in range(n):
        y=rng.integers(0,P*8-P); x=rng.integers(0,P*8-P)
        p=grande[y:y+P, x:x+P].copy()
        if tincion is not None:
            p = np.clip(p*np.array(tincion,np.float32), 0, 1)
        X[k]=(p*255).astype(np.uint8)
    return X

def conjunto(n_por_clase=300, semilla=7, tincion=None, P=96):
    sano  = lote(n_por_clase, 1.0, semilla,   P=P, tincion=tincion)
    tumor = lote(n_por_clase, 3.1, semilla+1, P=P, tincion=tincion)
    X=np.concatenate([sano,tumor]); y=np.r_[np.zeros(n_por_clase,int), np.ones(n_por_clase,int)]
    return X,y
