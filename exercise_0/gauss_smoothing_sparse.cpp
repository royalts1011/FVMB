

#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace torch::indexing;

namespace F = torch::nn::functional;


int main() {
    timeval time1,time2;

    torch::jit::script::Module tensors = torch::jit::load("../../ct_image_jit.pth");
    torch::Tensor img = tensors.attr("img_pytorch").toTensor();
    int B = img.size(0); int C = img.size(1); int H = img.size(2); int W = img.size(3);
    std::cout << "size (img) " << B << "x" << C << "x" << H << "x" << W <<"\n";


    gettimeofday(&time1, NULL);
    //define filter coefficients
    auto filter = torch::exp(-(torch::pow(torch::linspace(-1, 1, 7, torch::kFloat64), 2)));
    filter /= torch::sum(filter);
    std::cout << filter << "\n";

    //create a sparse kernel matrix
    auto xy = torch::arange(H*W);
    auto A_x = torch::sparse_coo_tensor(torch::stack({xy, xy}, 0), torch::zeros_like(xy, torch::kFloat64), {H*W,H*W});
    auto A_y = torch::sparse_coo_tensor(torch::stack({xy, xy}, 0), torch::zeros_like(xy, torch::kFloat64), {H*W,H*W});
    xy = xy.view({H, W});


    for(int i=0; i<4; i++){
        auto conv_x = torch::stack({xy.index({Slice(i, None), Slice()}), xy.index({Slice(None, H - i), Slice()})}, 0).view({2, -1});
        auto conv_y = torch::stack({xy.index({Slice(), Slice(i, None)}), xy.index({Slice(), Slice(None, W - i)})}, 0).view({2, -1});
        A_x += torch::sparse_coo_tensor(conv_x, torch::ones(conv_x.size(1)) * filter.index({i + 3}), {H*W,H*W});
        A_y += torch::sparse_coo_tensor(conv_y, torch::ones(conv_y.size(1)) * filter.index({i + 3}), {H*W,H*W});


        if(i>0){
            A_x += torch::sparse_coo_tensor(conv_x.flip(0), torch::ones(conv_x.size(1)) * filter.index({i + 3}), {H*W,H*W});
            A_y += torch::sparse_coo_tensor(conv_y.flip(0), torch::ones(conv_y.size(1)) * filter.index({i + 3}), {H*W,H*W});
        }

    }
    std::cout << A_x._nnz() << " " << A_y._nnz()<<"\n";

    //sparse matrix multiplication
    auto smoothed_img = torch::_sparse_mm(A_y, torch::_sparse_mm(A_x, img.to(torch::kFloat64).reshape({-1, 1}))).view({H, W});

    //print time
    gettimeofday(&time2, NULL);
    float timeP=1.0f*(time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6));
    std::cout << "runtime "<<timeP<<" secs\n";

    //save image
    auto bytes = torch::jit::pickle_save(smoothed_img.view({H,W}));
    std::ofstream fout("../output.pth", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();

    return 0;
}