import torch
from matplotlib import pyplot as plt

img = torch.load('./ct_image_pytorch.pth')
_, _, h, w = img.shape
matrix_shape = (h * w, h * w)

# task 1
filter = torch.exp(-(torch.linspace(-1, 1, 7, dtype=torch.float64) ** 2))
filter /= filter.sum()
print(filter)

xy = torch.arange(h * w)
A_x = torch.sparse_coo_tensor(torch.stack([xy, xy], 0), torch.zeros_like(xy, dtype=torch.float64), matrix_shape)
A_y = torch.sparse_coo_tensor(torch.stack([xy, xy], 0), torch.zeros_like(xy, dtype=torch.float64), matrix_shape)
xy = xy.view(h, w)
for i in range(4):
    conv_x = torch.stack([xy[i:, :], xy[:h - i, :]], 0).view(2, -1)
    conv_y = torch.stack([xy[:, i:], xy[:, :w - i]], 0).view(2, -1)
    A_x += torch.sparse_coo_tensor(conv_x, torch.ones(conv_x.shape[1]) * filter[i + 3], matrix_shape)
    A_y += torch.sparse_coo_tensor(conv_y, torch.ones(conv_y.shape[1]) * filter[i + 3], matrix_shape)
    if i > 0:
        A_x += torch.sparse_coo_tensor(conv_x.flip(0), torch.ones(conv_x.shape[1]) * filter[i + 3], matrix_shape)
        A_y += torch.sparse_coo_tensor(conv_y.flip(0), torch.ones(conv_y.shape[1]) * filter[i + 3], matrix_shape)

print(A_x._nnz(), A_y._nnz())

smoothed_img = torch._sparse_mm(A_y, torch._sparse_mm(A_x, img.double().reshape(-1, 1))).view(h, w)

plt.imshow(img.squeeze().data, 'gray')
plt.figure()
plt.imshow(smoothed_img.data, 'gray')
plt.show()

torch.save(smoothed_img, './output.pth')
