#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include "Net.cpp"

using namespace torch::indexing;
namespace F = torch::nn::functional;

// function declarations (definitions at the end of file)
torch::Tensor gen_heatmaps(torch::Tensor offset);
torch::Tensor gen_patches(torch::Tensor imgs, torch::Tensor init_pos);
torch::Tensor gen_eval_grids(torch::Tensor position);
torch::Tensor own_eval(torch::Tensor pred_heatmaps, torch::Tensor target_grids);

int main() {
    timeval time1, time2;
    //load data
    torch::jit::script::Module tensors = torch::jit::load("../fvmbv_exercise6_jit.pth");
    torch::Tensor img_train = tensors.attr("img_train").toTensor();
    torch::Tensor img_test = tensors.attr("img_test").toTensor();
    torch::Tensor pts_train = tensors.attr("pts_train").toTensor();
    torch::Tensor pts_test = tensors.attr("pts_test").toTensor();
    torch::Tensor ground_truth = tensors.attr("ground_truth").toTensor();

    auto coordinates_train = (pts_train + 1) * torch::tensor({11, 319}).view({1, 1, 1, 2}) / 2;
    auto coordinates_test = (pts_test + 1) * torch::tensor({11, 319}).view({1, 1, 1, 2}) / 2;

    std::cout << "coordinates_train:" << coordinates_train.sizes() << std::endl;
    std::cout << "img_test:" << img_test.sizes() << std::endl;
    std::cout << "pts_train:" << pts_train.sizes() << std::endl;

    //todo: provide names and student nr_s
    std::cout << "Solution by : Konrad v. Kuegelgen (676609), Falco Lentzsch (685454), Jesse Kruse (675710) \n";

    //your code here

    // Init the net
    int64_t numLandmarks = 25;
    auto net = ConvNet(numLandmarks);

    auto loss_func = torch::nn::MSELoss();
    auto optimizer = torch::optim::Adam(net.parameters(), 0.015);
    int batch_size = 4;
    int num_epoches = 300;

    float loss_epoch;


    // ===== Training loop =====

    for(int i=0; i < num_epoches; i++)  {
        optimizer.zero_grad();

        auto idx_epoch = torch::randperm(9).index({Slice({0,8})}).view({-1, batch_size});

        for(int j=0; j < idx_epoch.size(0); j++)    {

            auto idx = idx_epoch.index({j});

            auto img_batch = img_train.index({idx, Slice(), Slice(), Slice()});
            auto pts_batch = pts_train.index({idx, Slice(), Slice(), Slice()});

            auto not_idx = torch::ones({9});
            for(int k=0; k < idx.size(0); k++)  {
                not_idx[idx.index({k})] = 0;
            }
            auto not_idx_short = not_idx.nonzero().flatten();
            not_idx_short = not_idx_short.index({torch::randperm(5).index({Slice({0,-1})})});

            auto offset = pts_batch - pts_train.index({not_idx_short});

            auto heatmap_batch = gen_heatmaps(offset);
            auto patch_batch = gen_patches(img_batch, pts_train.index({not_idx_short}) );

            auto out_heatmaps = net.forward(patch_batch);
            auto loss = loss_func(out_heatmaps, heatmap_batch);

            loss_epoch = loss.item().toFloat();
            loss.backward();
        }

        optimizer.step();

        std::cout << "Epoch: " << i << " | Loss: " << loss_epoch << std::endl;
    }


    // ====== Evaluation ======
    
    net.eval();

    auto patch_batch = gen_patches(img_test.expand({9,1,320,312}), pts_train);
    auto out_heatmaps = net.forward(patch_batch);
    auto eval_grids = gen_eval_grids(pts_train);

    auto pred_coordinates = own_eval(out_heatmaps, eval_grids);

    // write tensor for visualization
    auto bytes = torch::jit::pickle_save(pred_coordinates);
    std::ofstream fout("../predicted_coordinates.pth", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
    std::cout << "Export finished... Can be visualized with provided the Python script." << std::endl;
}


// ===== Function Definitions =====

/**
 * For given offsets the function returns the heatmap where the center of the patch should be located.
 * @param offset
 * @return tensor with heatmaps for given offsets
 */
torch::Tensor gen_heatmaps(torch::Tensor offset) {
    auto xy = torch::linspace(-0.3, 0.3, 17);
    auto grids = torch::meshgrid({xy, xy});
    auto x_grid = grids[0];
    auto y_grid = grids[1];
    auto xy_grid = torch::cat({x_grid.unsqueeze(2), y_grid.unsqueeze(2)}, 2).view({1, 1, -1, 2});

    auto target_map = 20 * torch::exp(-40 * ((xy_grid - offset).pow(2)).sum(-1));

    return target_map.view({4, 25, 17, 17});
}

/**
 * Generates image patches 23x23 for the given init positions
 * @param imgs
 * @param init_pos
 * @return tensor with img patches
 */
torch::Tensor gen_patches(torch::Tensor imgs, torch::Tensor init_pos) {
    auto xy = torch::linspace(-0.4, 0.4, 23);
    auto grids = torch::meshgrid({xy, xy});
    auto x_grid = grids[0];
    auto y_grid = grids[1];
    auto xy_grid = torch::cat({x_grid.unsqueeze(2), y_grid.unsqueeze(2)}, 2).expand({25, 23, 23, 2});

    init_pos = init_pos.unsqueeze(2).expand({init_pos.size(0), 25, 23, 23, 2});

    auto patches = torch::zeros({imgs.size(0), 25, 23, 23});

    for (int i = 0; i < imgs.size(0); i++) {
        auto tmp_grid = xy_grid + init_pos.index({i});

        auto grid_sample_options = F::GridSampleFuncOptions().align_corners(false);
        auto tmp_patch = F::grid_sample(imgs.index({i}).unsqueeze(0).expand({25, 1, 320, 312}), tmp_grid, grid_sample_options).squeeze();
        patches.index({i}) = tmp_patch;
    }
    return patches;

}
/**
 * generates grids for the evaluation function. 17x17 grids [-0.3,0.3] + offset for position in the image.
 * @param position
 * @return grid tensors
 */
torch::Tensor gen_eval_grids(torch::Tensor position) {
    auto xy = torch::linspace(-0.3, 0.3, 17);
    auto grids = torch::meshgrid({xy, xy});
    auto x_grid = grids[0];
    auto y_grid = grids[1];
    auto xy_grid =
            torch::cat({x_grid.unsqueeze(2), y_grid.unsqueeze(2)}, 2).view({1, 1, -1, 2}).expand({9, 25, 289, 2}) +
            position;

    return xy_grid;
}
/**
 * evaluation function that estimates the positions of landmarks by conerting the heatmaps to point positions [-1, 1],
 * averaging over all of them and converting them into the image coordinate system.
 * @param pred_heatmaps
 * @param target_grids
 * @return predictions for the positions of the landmarks
 */
torch::Tensor own_eval(torch::Tensor pred_heatmaps, torch::Tensor target_grids) {
    auto predicted_pts = torch::zeros({9,25,2});
    auto max_pos = torch::argmax(pred_heatmaps.view({9,25,-1}), 2);

    for(int i=0; i < 9; i++)  {
        for(int j=0; j < 25; j++) {
            predicted_pts.index({i,j}) = target_grids.index({i,j,max_pos.index({i,j}), Slice()});
        }
    }
    predicted_pts = predicted_pts.mean(0);
    auto pred_coordinates = (predicted_pts+1) * torch::tensor({312-1, 320-1}) / 2;

    return pred_coordinates;
}



