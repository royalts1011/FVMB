import torch
from matplotlib import pyplot as plt

img = torch.load('./ct_image_pytorch.pth')

filter_kernel = torch.exp(-(pow(torch.linspace(-1, 1, 7),2)))
filter_kernel /= filter_kernel.sum()

'''
- Build Matrix shape
- use manuel Parameter m = 2 or 3 and n = 5 or 7 to visualize
- xy ist Matrix zu Bild equivalenter Größe
'''
_, _, m, n = img.shape
#m = 2
#n = 4
size = (m * n, m * n)
xy = torch.arange(m * n)

'''
Hier wird die Sparse-Matrix für die Filterung in x und y Richtung initialisiert
Normal geben wir hier folgende Parameter an:
1. idx = torch.stack([xy, xy], dim=0)
2. val = torch.zeros_like(xy, dtype=torch.float64)
3. shape(size)
'''
A_x = torch.sparse_coo_tensor(size)
A_y = torch.sparse_coo_tensor(size)

# Um die Matrix zu betrachten
#print(A_x.to_dense())



# xy wieder zurück bauen zu Bildgröße damit wir daraus
# besser die Indizes der benachbarten Pixel ablesen können.
xy = xy.view(m,n)


'''
Stellen wir uns nun xy als das Bild vor, was nun nur die Indizes enthält.
Die Schleife sorgt nun dafür, das Pixel, die eine "kante" zueinander haben wir einen
Eintrag in die Sparse Matritzen A_x und A-y machen.
0: Bildet die Diagonale(rot), da jeder Pixel zu sich selnst erst einmal eine Kante hat.
D.h. 0-0, 1-1,....

Danach werden unabhängig in x und y-Richtung die Paramter ermittelt in Abhängigkeit der Kanten
Da diese nun in beide Richtungen Gelden, da wenn a-b gilt auch b-a gilt, müssen wir die
ermittelten Paramter noch an den invertierten Positionen eintragen.

- coefficient beinhaltet immer die Indizes durch stacken zweier Matrixen
- coefficients_x verliert immer eine Zeile 
- coefficients_y verliert immer eine Spalte 
'''
for i in range(4):
    # Zur Visualisierung der Indizes einzeln
    #print(xy[i:, :])
    #print(xy[:m - i, :])
    coefficients_x = torch.stack([xy[i:, :], xy[:m - i, :]], 0).view(2,-1)
    coefficients_y = torch.stack([xy[:, i:], xy[:, :n - i]], 0).view(2,-1)
    # Zur Visualisierung der Indizes in der matrix
    #print(coefficients_x)
    A_x += torch.sparse_coo_tensor(coefficients_x, torch.ones(coefficients_x.shape[1]) * filter_kernel[3 + i], size)
    A_y += torch.sparse_coo_tensor(coefficients_y, torch.ones(coefficients_y.shape[1]) * filter_kernel[3 + i], size)
    if i > 0:
        A_x += torch.sparse_coo_tensor(coefficients_x.flip(0), torch.ones(coefficients_x.shape[1]) * filter_kernel[i + 3], size)
        A_y += torch.sparse_coo_tensor(coefficients_y.flip(0), torch.ones(coefficients_y.shape[1]) * filter_kernel[i + 3], size)


print(A_x._nnz())
print(A_y._nnz())
#print(A_x.to_dense())

'''
Bild zu Vektor umbauen und Filter anwenden.
'''
# Macht aus dem Bild 320x312 einen Vektor 98840 x 1
flat_img= img.reshape(-1, 1)

filtered_img = torch._sparse_mm(A_x,flat_img)
filtered_img = torch._sparse_mm(A_y,filtered_img)
filtered_img = filtered_img.view(m,n)
#test = torch._sparse_mm(A_x, img.double().reshape(-1, 1))

plt.imshow(filtered_img.data, 'gray')
plt.show()

torch.save(filtered_img, './output.pth')