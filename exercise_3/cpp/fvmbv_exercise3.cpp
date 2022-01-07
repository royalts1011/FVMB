#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace torch::indexing;
namespace F = torch::nn::functional;


int main() {
    timeval time1,time2;
    std::cout << "Name, Number \n";
    torch::jit::script::Module tensors = torch::jit::load("../exercise3_stereo_jit.pth");
    torch::Tensor fixed = tensors.attr("fixed").toTensor();
    torch::Tensor moving = tensors.attr("moving").toTensor();
    torch::Tensor data = tensors.attr("data").toTensor();
    int H = fixed.size(0);
    int W = fixed.size(1);
    torch::Tensor output = fixed;
    std::cout << fixed.sizes() << "\n";
    gettimeofday(&time1, NULL);

    //your code here

    // Task 1
    int K = 67;
    auto beta = torch::tensor(0.11, torch::kFloat32);
    auto gamma = torch::tensor(7, torch::kFloat32);
    auto delta = torch::tensor(2, torch::kFloat32);

    auto weight = torch::linspace(-0.5, 0.5, 3);
    auto convOptions = F::Conv2dFuncOptions().padding({0,1});
    auto moving_x = F::conv2d(moving.view({1,1,H,W}), weight.view({1,1,1,3}), convOptions);
    auto fixed_x = F::conv2d(fixed.view({1,1,H,W}), weight.view({1,1,1,3}), convOptions);

    auto padOptions = F::PadFuncOptions({66,0});
    auto moving_pad = F::pad(moving, padOptions).view({1,1,H, W + K - 1});
    auto moving_x_pad = F::pad(moving_x, padOptions).view({1,1,H, W + K -1});

    auto unfoldOptions = F::UnfoldFuncOptions({1, 67});
    auto M = F::unfold(moving_pad, unfoldOptions).view({1,K, H, W});
    auto Mx = F::unfold(moving_x_pad, unfoldOptions).view({1,K, H, W});

    auto dif = torch::abs(fixed - M);
    auto dif_x = torch::abs(fixed_x - Mx);

    auto Dy = torch::add(torch::mul(beta,torch::where(dif < gamma, dif, gamma)), torch::mul((1-beta), torch::where(dif_x < delta, dif_x, delta)));
    Dy = Dy.squeeze();


    // Task 2
    int K_bsp = 5;
    int N_bsp = 4;

    auto idx_K = torch::arange(K_bsp).unsqueeze(0);
    auto idx_K_rev = idx_K.transpose(1,0);
    auto regular = torch::abs(idx_K - idx_K_rev);

    // forward path:
    auto forward = torch::zeros_like(data);
    forward.index({Slice(0,None), Slice()}) = data.index({Slice(0,None), Slice()});
    auto b = torch::zeros({N_bsp-1, K_bsp});


    for(int i=0; i<3; i++)  {
        auto all_costs  = forward.index({Slice(i,i+1), Slice()}).repeat({K_bsp, 1}) + regular;
        //std::cout << all_costs.sizes() << std::endl;
        auto costs = std::get<0>(torch::min(all_costs, 1));
        auto b_i = std::get<1>(torch::min(all_costs, 1));
        forward.index({Slice(i+1, None), Slice()}) = data.index({Slice(i+1,None), Slice()}) + costs;
        b.index({Slice(i, None), Slice()}) = b_i;
    }

    // backward path:
    auto backward = torch::zeros_like(data);
    backward.index({Slice(-1, None), Slice()}) = data.index({Slice(-1, None), Slice()});

    for(int i=2; i>=0; i--)   {
        auto all_costs = backward.index({Slice(i+1, i+2), Slice()}).repeat({K_bsp, 1}) + regular;
        auto costs = std::get<0>(torch::min(all_costs, 1));
        backward.index({Slice(i, i+1), Slice()}) = data.index({Slice(i, i+1), Slice()}) + costs;
    }

    auto comb_costs = forward + backward - data;

    std::cout << std::endl;
    std::cout << "In the following the first sequence is the first row, second sequence the second and so on... " << std::endl;
    std::cout << std::endl;
    std::cout << "Forward path:" << std::endl;
    std::cout << forward << std::endl;
    std::cout << std::endl;
    std::cout << "paths: (Because of zero indexing add 1 to get the table from VL)" << std::endl;
    std::cout << b << std::endl;
    std::cout << std::endl;
    std::cout << "Combined Costs:" << std::endl;
    std::cout << comb_costs <<std::endl;
    std::cout << std::endl;


    // Task 3
    idx_K = torch::arange(K, torch::kFloat32).unsqueeze(0).repeat({K,1});
    idx_K_rev = idx_K.transpose(1,0);
    regular = torch::abs(idx_K - idx_K_rev);
    regular = torch::where(regular <= torch::tensor(3.5, torch::kFloat), regular, torch::tensor(3.5, torch::kFloat));
    regular = regular.unsqueeze(2).repeat({1,1,H});

    // forward path:
    forward = torch::zeros_like(Dy);
    forward.index({Slice(), Slice(), Slice(0,1)}) = Dy.index({Slice(), Slice(), Slice(0,1)});
    b = torch::zeros({K,H,W});

    for(int i=0; i<W-1; i++)    {
        auto all_costs = forward.index({Slice(), Slice(), Slice(i, i+1)}).squeeze().repeat({K,1,1}) + regular;
        auto costs = std::get<0>(torch::min(all_costs, 1));
        forward.index({Slice(), Slice(), Slice(i+1, i+2)}) = (Dy.index({Slice(), Slice(), Slice(i+1, i+2)}).squeeze() + costs).unsqueeze(2);
    }

    // backward path:
    backward = torch::zeros_like(Dy);
    backward.index({Slice(), Slice(), Slice(-1,None)}) = Dy.index({Slice(), Slice(), Slice(-1,None)});

    for(int i=W-2; i>=0; i--)    {
        auto all_costs = backward.index({Slice(), Slice(), Slice(i+1, i+2)}).squeeze().repeat({K,1,1}) + regular;
        auto costs = std::get<0>(torch::min(all_costs, 1));
        backward.index({Slice(), Slice(), Slice(i, i+1)}) = (Dy.index({Slice(), Slice(), Slice(i, i+1)}).squeeze() + costs).unsqueeze(2);
    }

    comb_costs = forward + backward - Dy;
    output = - torch::argmin(comb_costs, 0);


    gettimeofday(&time2, NULL);

    float timeP=1.0f*(time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6));

    std::cout << "runtime "<<timeP<<" secs\n";
    auto bytes = torch::jit::pickle_save(output.to(torch::kFloat32));
    std::ofstream fout("../stereo_output.pth", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();

}


