from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def entropy(signal):
        len=signal.size
        signal_val=list(set(signal))
        propabilities=[np.size(signal[signal==i])/(1.0*len) for i in signal_val]
        entropy=np.sum([-1.0*p*np.log2(p) for p in propabilities])
        return entropy

colorIm=Image.open('wall.png')
greyIm=colorIm.convert('L')
colorIm=np.array(colorIm)
greyIm=np.array(greyIm)

N=3
S=greyIm.shape
Entropy=np.array(greyIm)
for row in range(S[0]):
        for col in range(S[1]):
                Lx=np.max([0,col-N])
                Ux=np.min([S[1],col+N])
                Ly=np.max([0,row-N])
                Uy=np.min([S[0],row+N])
                region=greyIm[Ly:Uy,Lx:Ux].flatten()
                Entropy[row,col]=entropy(region)

plt.subplot(1,3,1)
plt.imshow(colorIm)

plt.subplot(1,3,2)
plt.imshow(greyIm, cmap=plt.cm.gray)

plt.subplot(1,3,3)
plt.imshow(Entropy, cmap=plt.cm.jet)
plt.xlabel('Entropy in 6x6 neighbourhood')
plt.colorbar()

plt.show()