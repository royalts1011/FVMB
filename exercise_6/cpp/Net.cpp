//
// Created by Jesse Kruse on 26.01.22.
//
#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace torch::indexing;
namespace F = torch::nn::functional;

struct ConvNet : torch::nn::Module {
    explicit ConvNet(int64_t num_landmarks) {
        layer1 = register_module("layer1",torch::nn::Conv2d(torch::nn::Conv2dOptions(num_landmarks,64,5).bias(false)));
        batch1 = register_module("batch1", torch::nn::BatchNorm2d(64));
        layer2 = register_module("layer2",torch::nn::Conv2d(torch::nn::Conv2dOptions(64,64,3).bias(false)));
        batch2 = register_module("batch2", torch::nn::BatchNorm2d(64));
        layer3 = register_module("layer3",torch::nn::Conv2d(torch::nn::Conv2dOptions(64,64,3).bias(false)));
        batch3 = register_module("batch3", torch::nn::BatchNorm2d(64));

        layer4 = register_module("layer4",torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64,64,5).bias(false)));
        batch4 = register_module("batch4", torch::nn::BatchNorm2d(64));
        layer5 = register_module("layer5",torch::nn::Conv2d(torch::nn::Conv2dOptions(64,25,3).bias(true)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = F::relu(batch1(layer1(x)));
        x = F::relu(batch2(layer2(x)));
        x = F::relu(batch3(layer3(x)));

        x = F::relu(batch4(layer4(x)));
        x = F::relu(layer5(x));

        return x;
    }
    // Nullptr noch nötig?
    torch::nn::Conv2d layer1{nullptr};
    torch::nn::BatchNorm2d batch1{nullptr};
    torch::nn::Conv2d layer2{nullptr};
    torch::nn::BatchNorm2d batch2{nullptr};
    torch::nn::Conv2d layer3{nullptr};
    torch::nn::BatchNorm2d batch3{nullptr};

    torch::nn::ConvTranspose2d layer4{nullptr};
    torch::nn::BatchNorm2d batch4{nullptr};
    torch::nn::Conv2d layer5{nullptr};
};

