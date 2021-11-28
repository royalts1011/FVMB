#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace torch::indexing;

namespace F = torch::nn::functional;

struct DeformableNet : torch::nn::Module {
    DeformableNet() {
        // the FC1 Conv1d module contains the displacements to be optimised
        fc1 = register_module("fc1", torch::nn::Conv1d(51, 2, 38));
    }

    torch::Tensor forward() {
        //todo: extend the B-spline function to contain three 2D avg_pooling layers with kernel = 5, stride = 1 and padding = 2
        auto grid = fc1->weight.unsqueeze(0);
        grid = F::avg_pool2d(grid,F::AvgPool2dFuncOptions(5).stride(1).padding(2));
        grid = F::avg_pool2d(grid,F::AvgPool2dFuncOptions(5).stride(1).padding(2));
        grid = F::avg_pool2d(grid,F::AvgPool2dFuncOptions(5).stride(1).padding(2));
        // bis hier B-spline jetzt folgt das upsampling
        grid = F::interpolate(grid,F::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(true)
                                                .scale_factor(std::vector<double> {6.0, 6.0}));


        return grid;
    }

    // Use one of many "standard library" modules.
    torch::nn::Conv1d fc1{nullptr};
};

int main() {
    timeval time1, time2;
    // Hier wurde glaube ich fälschlicher weise das images_contrast_jit.pth geladen!!! Mit images_jit.pth geht es jetzt.
    torch::jit::script::Module tensors = torch::jit::load("../images_jit.pth");
    torch::Tensor fixed = tensors.attr("fixed").toTensor();
    torch::Tensor moving = tensors.attr("moving").toTensor();
    int H = fixed.size(2);
    int W = fixed.size(3);


    //todo: Print out Name and student-nr
    std::cout << "Solution by: Falco Lentzsch(685454),Konrad von Kuegelgen, Jesse Kruse(675710) \n";

    gettimeofday(&time1, NULL);
    auto net = std::make_shared<DeformableNet>();
    //initialise parameters with zeros!
    torch::nn::init::zeros_(net->fc1->weight);

    //todo: Create identity grid
    auto identity_grid = F::affine_grid(torch::eye(2, 3).unsqueeze(0), {1, 1, H, W}, true);
    // Learningrate hier angepasst auf 0.015 wie in Python script
    torch::optim::Adam optimizer(net->parameters(), 0.015);
    auto fine_grid = torch::zeros({1,2,H,W});

    //given: weights, mean and variance for fixed image
    int K = 5; int kw = (K-1)/2;
    auto avg_kernel = F::AvgPool2dFuncOptions(K).stride(1).padding(kw);
    auto normalise = F::avg_pool2d(torch::ones_like(moving), avg_kernel);
    auto fixed_mean = F::avg_pool2d(fixed, avg_kernel)/normalise;
    auto fixed_var = F::avg_pool2d(fixed.pow(2), avg_kernel)/normalise-fixed_mean.pow(2);


    //given: Optimisation loop
    for(int i=0; i<100; i++){

        //todo: Reset gradients.
        optimizer.zero_grad();

        //Execute the model to obtain displacements.
        fine_grid = net->forward();

        //todo: Warp image
        auto mapping_grid = identity_grid + fine_grid.permute({0,2,3,1});
        auto warped =  F::grid_sample(moving, mapping_grid,F::GridSampleFuncOptions().align_corners(true));

        //todo: Compute Mean Squared Error for loss criteria --> Option muss hier angegeben werden um die Summe zu Bilden
        auto loss = F::mse_loss(warped, fixed, F::MSELossFuncOptions(torch::kSum));

        //todo: (Bonus) Compute Normalised Cross Correlation for loss criteria
        // ---> Wurde nur in Python umgesetzt <---

        // ---> Hier wird der Loss ausgegeben <---
        std::cout << "NCC: " << loss << "\n";

        //todo: Compute gradients of the loss w.r.t. the parameters of our model (-> backward).
        loss.backward();

        //todo: Update the parameters based on the calculated gradients.
        optimizer.step();

    }
    gettimeofday(&time2, NULL);

    float timeP=1.0f*(time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6));

    std::cout << "runtime "<<timeP<<" secs\n";
    auto bytes = torch::jit::pickle_save(fine_grid.view({2,H,W}));

    std::ofstream fout("../adam_flow.pth", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}
