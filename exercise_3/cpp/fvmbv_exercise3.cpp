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



    gettimeofday(&time2, NULL);

    float timeP=1.0f*(time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6));

    std::cout << "runtime "<<timeP<<" secs\n";
    auto bytes = torch::jit::pickle_save(output.to(torch::kFloat32));
    std::ofstream fout("../stereo_output.pth", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();

}


