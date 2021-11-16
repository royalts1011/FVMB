import torch

# given: function for conjugate gradient
def sparseCG(A,b,iterations): #A sparse matrix, b dense vector
#conjugate gradient https://william-dawson.github.io/blog/method/2017/10/01/matrixcg.html
    x = torch.zeros(b.numel(),1).to(A.device)
    r = b.view(-1,1) - torch.spmm(A,x)
    p = r.clone()
    for i in range(iterations):
        Ap = torch.spmm(A,p)
        top = (r*r).sum()
        bottom = (p*Ap).sum()
        alpha = top / (bottom+0.0001)
        x = x + alpha * p
        r = r - alpha * Ap

        new_top = (r*r).sum()
        beta = new_top/(top+0.0001)
        p = r + beta * p
    return x