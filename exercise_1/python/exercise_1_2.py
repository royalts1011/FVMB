import torch
from matplotlib import pyplot as plt

data = torch.load('graph_s_data_pytorch.pth')
x = data['x']
y = data['y']
values = data['values']
n = x.size(0)
xy = torch.stack((x,y),1)

order_noise = torch.zeros(n)
order_noise[torch.sort(values)[1]] = torch.linspace(0,1,n)

# todo: compute distances of xy-coordinates
distance_matrix = (xy.unsqueeze(1) - xy.unsqueeze(0)).pow(2).sum(2)

# todo: find the 16th smallest value per row/col using topk why second dim its symetric?

kNN_values_dim1, kNN_indices_dim1 = torch.topk(distance_matrix, k=17, dim=0, largest=False)
# Removing Distances like 1 to 1 so the Diagonal for IDX and Value
kNN_indices_dim1 = kNN_indices_dim1[1:, :]
kNN_values_dim1 = kNN_values_dim1[1:, :]
weights_dim1 = -torch.exp(-2 * kNN_values_dim1)

# todo Second Directin not needed cause the matrix is symetric
# kNN_values_dim2, kNN_indices_dim2 = torch.topk(distance_matrix, k=17 , dim=1, largest=False)
# Removing Distances like 1 to 1 so the Diagonal for IDX and Value, also in second dim
# kNN_indices_dim2 = kNN_indices_dim2[:, 1:]
# kNN_values_dim2 = kNN_values_dim2[:, 1:]
# weights_dim2 = -torch.exp(-2 * kNN_values_dim2)


print("Index in first DIM: \n ", kNN_indices_dim1[0][6])
print("Value in first DIM: \n ", kNN_values_dim1[0][6])
#print("Index in second DIM: \n ", kNN_indices_dim2[6][0])
#print("Value in second DIM: \n ", kNN_values_dim2[6][0])
L = torch.zeros_like(distance_matrix)
L.scatter_(dim=0, index=kNN_indices_dim1, src=weights_dim1)
#L.scatter(dim=1, index=kNN_indices_dim2, src=weights_dim2)
print("Entry in first DIM ind Matrix: \n ", L[0][7])
print("Entry in second DIM ind Matrix: \n ", L[7][0])

# Diagonale und Diagonalmatrix bauen und zu Matrix L dazu addieren
diagonal = - torch.sum(L, dim=1)
diagonal_matrix = torch.diag(input=diagonal)
L += diagonal_matrix



# given: solve for denoised values on graph
values_solve = torch.solve(values.reshape(-1,1),L*25+torch.eye(n))
values_solve = values_solve.solution
order_solve = torch.zeros(n)
order_solve[torch.sort(values_solve.squeeze(1))[1]] = torch.linspace(0,1,n)

# given: plot result
cm = plt.cm.get_cmap('jet')
plt.subplot(121), plt.axis('equal'), plt.title('noisy')
plt.scatter(x,y,c=order_noise.numpy(),vmin=0,vmax=1,s=35,cmap=cm)
plt.subplot(122), plt.axis('equal'), plt.title('denoised')
plt.scatter(x,y,c=order_solve.numpy(),vmin=0,vmax=1,s=35,cmap=cm)

#plt.savefig('plot.png')
plt.show()

print("debug")
