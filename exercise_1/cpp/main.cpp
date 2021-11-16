#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

using namespace torch::indexing;

namespace F = torch::nn::functional;

torch::Tensor sparseCG(torch::Tensor A, torch::Tensor b, int iterations) {
    b = b.view({-1, 1});
    auto x = torch::zeros_like(b);
    auto r = b.view({-1, 1}) - torch::_sparse_mm(A, x);
    auto p = r.clone();
    for (int i = 0; i < iterations; i++) {
        auto Ap = torch::_sparse_mm(A, p);
        auto top = (r * r).sum();
        auto bottom = (p * Ap).sum();
        auto alpha = top / (bottom + 1e-9);
        x = x + alpha * p;
        r = r - alpha * Ap;
        auto new_top = (r * r).sum();
        auto beta = new_top / (top + 1e-9);
        p = r + beta * p;
    }
    return x;
}


int main() {

    //Exercise 1, Task 1 edge-preserving denoising on regular image grid
    torch::jit::script::Module images = torch::jit::load("ct_image_jit.pth");
    torch::Tensor img = images.attr("img_pytorch").toTensor().squeeze();
    std::cout << "size (img) " << img.size(0) << " x " << img.size(1) << "\n";
    int H = img.size(0); int W = img.size(1);
    float lambda_ = 200; float sigma2 = 0.06*0.06;
    //todo: create range tensor xy0 and sliced tensors xy1 and xy2 and index tensor ii

    //todo: compute edge preserving weights

    //todo: compute first sparse matrix of size H*W,H*W with the wights calculated above

    //todo: flip the direction of the edges and add a second sparse matrix to the initial one

    //todo: create and add sparse matrices by slicing xy in the first dimension

    //todo: use torch::_sparse_sum to compute diagonal elements

    //todo: Create Laplacian matrix L with positive main diagonal elements (plus identity!) and negative weights


    //given: solve equation system
    auto denoised_img = sparseCG(L,img.reshape({-1,1}),150);

    //given: save denoised image
    auto bytes_img = torch::jit::pickle_save(denoised_img.reshape({H,W}));
    std::ofstream fout_img("img_denoise_output.pth", std::ios::out | std::ios::binary);
    fout_img.write(bytes_img.data(), bytes_img.size());
    fout_img.close();

    //Exercise 1, Task 2 denoising on irregular graph
    //given: read tensors from jit.pth file
    torch::jit::script::Module tensors = torch::jit::load("graph_s_data_jit.pth");
    torch::Tensor x = tensors.attr("x").toTensor();
    torch::Tensor y = tensors.attr("y").toTensor();
    torch::Tensor values = tensors.attr("values").toTensor();

    //std::cout << "size (x) " << x.size(0) <<"\n";

    auto xy = torch::stack({x,y},1);
    int n = x.size(0);

    //todo: compute distances of xy-coordinates


    //todo: find the 16th smallest value per row/col using topk and std::get<0>


    //todo: construct laplace matrix (dense is fine)


    //given: solve for denoised values on graph
    auto value_solve = std::get<0>(torch::solve(values.reshape({-1,1}),laplace*25+torch::eye(n)));

    //given: pickle result tensor and write to file
    auto bytes = torch::jit::pickle_save(value_solve.reshape(-1));
    std::ofstream fout("graph_s_output.pth", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();


}
