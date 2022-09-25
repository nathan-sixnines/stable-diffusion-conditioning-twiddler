from operator import index
import torch
import cv2
import numpy as np

c1 = torch.load("castle.npy")
c2 = torch.load("waterfall.npy")

c3 = c1 - c2

def to_img(vec, name):

    np_vec = np.squeeze(vec.cpu().numpy())
    ma = np.max(np_vec)
    mi = np.min(np_vec)

    spread = ma - mi

    np_vec += mi
    np_vec = np_vec / spread
    np_vec = (np_vec * 255).astype(np.uint8)

    print(np.max(np_vec))

    cv2.imwrite(name, np_vec)


print(torch.topk(c3,1))

to_img(c1,"c1.png")
to_img(c2,"c2.png")
to_img(c3,"c3.png")
print(c1.shape)

print(torch.max(c1))
print(torch.min(c1))

print(torch.max(c2))
print(torch.min(c2))

print(torch.max(c3))
print(torch.min(c3))

c3 = c3.cpu().numpy()

c3_shape = c3.shape

c3 = c3.reshape(-1)

result = {}

indexes = []

for i in range(20):
    print(f"iter {i=}")
    pos = np.argmax(c3)
    print(c3[pos])
    dex = np.unravel_index(pos,c3_shape)
    indexes.append(dex)
    print(c1[0,1,631])
    print(c2[0,1,631])
    result[np.unravel_index(pos,c3_shape)] = (float(c1[dex]),float(c2[dex]))

    c3[pos] = 0

print(result)

print(indexes)