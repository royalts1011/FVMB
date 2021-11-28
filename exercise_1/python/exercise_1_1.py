import torch
import utils
from matplotlib import pyplot as plt

img = torch.load('ct_image_pytorch.pth')
img_xy = img.reshape(-1)
_, _, m, n = img.shape


small_dim = True
if small_dim: m,n = 3,3
size = (m * n, m * n)

# Build Matrix containing indices with shape of image
xy = torch.arange(m * n).view(m,n)
if small_dim: print("Image corresponding index Matrix \n", xy)

# Index shifted indices for y-dim dont know
xy1_y = xy[:, 1:].reshape(-1)
xy2_y = xy[:, :-1].reshape(-1)
idx_y = torch.stack([xy1_y, xy2_y], 0)
if small_dim: print("y-Index-paires after shift \n", idx_y)


# create weights
lam = 200
sigma = 0.06
weights = - lam * torch.exp(- pow(sigma, 2) * torch.pow(img_xy[xy1_y] - img_xy[xy2_y], 2))

L = torch.sparse_coo_tensor(indices=idx_y, values=weights, size=size)
if small_dim: print("Dense matrix filled without flipped weights \n", L.to_dense())

L += torch.sparse_coo_tensor(indices=idx_y.flip(0), values=weights, size=size)
if small_dim: print("Dense matrix filled with all weights in y-neighbourhood \n", L.to_dense())

# Index shifted indices for x-dim dont know
xy1_x = xy[1:, :].reshape(-1)
xy2_x = xy[:-1, :].reshape(-1)
idx_x = torch.stack([xy1_x, xy2_x], 0)
if small_dim: print("x-Index-paires after shift \n", idx_x)

# create weights
weights = - lam * torch.exp(- pow(sigma, 2) * torch.pow(img_xy[xy1_x] - img_xy[xy2_x], 2))
L += torch.sparse_coo_tensor(indices=idx_x, values=weights, size=size)
L += torch.sparse_coo_tensor(indices=idx_x.flip(0), values=weights, size=size)
if small_dim: print("Dense matrix filled with all weights in x-neighbourhood \n", L.to_dense())


# build diagonal line by summing up line changing negativ foresign
diagonal = - torch.sparse.sum(L, dim=1).to_dense() + 1
diagonal_idx = torch.stack([xy.reshape(-1),xy.reshape(-1)],0)



# put diagonal in sparse matrix and add to L
diagonal_matrix = torch.sparse_coo_tensor(indices=diagonal_idx,values=diagonal,size=size)
L += diagonal_matrix
if small_dim: print("Full Matrix matrix \n", L.to_dense())


#print("debugg")
filtered_img = utils.sparseCG(L,img.reshape(-1,1),150).view(m,n)
#plt.imshow(filtered_img.data, 'gray')
#plt.show()

# given: visualize input, resulting images
plt.imshow(torch.clamp(img.reshape(-1,n).data,-500,500),'gray'), plt.colorbar()
plt.show()
plt.imshow(torch.clamp(filtered_img.reshape(-1,n).data,-500,500),'gray'), plt.colorbar()
plt.show()


torch.save(filtered_img, 'img_filtered_output.pth')