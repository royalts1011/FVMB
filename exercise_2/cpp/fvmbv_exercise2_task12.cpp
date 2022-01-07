#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace torch::indexing;

namespace F = torch::nn::functional;


//Given: Solver
torch::Tensor sparseCG(torch::Tensor A, torch::Tensor b, int iterations){
    b = b.view({-1,1});
    auto x = torch::zeros_like(b);
    auto r = b.view({-1,1}) - torch::_sparse_mm(A,x);
    auto p = r.clone();
    for(int i=0; i<iterations; i++){
        auto Ap = torch::_sparse_mm(A,p);
        auto top = (r*r).sum();
        auto bottom = (p*Ap).sum();
        auto alpha = top / (bottom+1e-9);
        x = x + alpha * p;
        r = r - alpha * Ap;
        auto new_top = (r*r).sum();
        auto beta = new_top/(top+1e-9);
        p = r + beta * p;
    }
    return x;
}
//given: Create an isotropic Laplace matrix
torch::Tensor laplaceMatrix(int H,int W){
    //define Laplace matrix for diffusion regularisation
    auto xy = torch::arange(H*W).view({H,W});
    //extract edges for four neighbours with slicing
    auto i1 = torch::cat({xy.index({Slice(1),Slice(None)}).reshape({1,-1}),
                          xy.index({Slice(None,-1),Slice(None)}).reshape({1,-1})},0);
    auto i2 = torch::cat({xy.index({Slice(None,-1),Slice(None)}).reshape({1,-1}),
                          xy.index({Slice(1,None),Slice(None)}).reshape({1,-1})},0);
    auto i3 = torch::cat({xy.index({Slice(None),Slice(1)}).reshape({1,-1}),
                          xy.index({Slice(None),Slice(None,-1)}).reshape({1,-1})},0);
    auto i4 = torch::cat({xy.index({Slice(None),Slice(None,-1)}).reshape({1,-1}),
                          xy.index({Slice(None),Slice(1,None)}).reshape({1,-1})},0);
    auto idx = torch::cat({i1,i2,i3,i4},1);

    //apply the hyperparameter lambda as 1250 (isotropic unit weighting)
    float lambda = 1250;
    auto val = torch::ones(2*(2*H*W-H-W));

    //compute the diagonal-sum and return the sparse matrix
    auto A = torch::sparse_coo_tensor(idx,-lambda*val,{H*W,H*W});
    auto D1 = -torch::_sparse_sum(A,0).to_dense();
    auto D = torch::sparse_coo_tensor(torch::cat({xy.view({1,-1}),xy.view({1,-1})},0),D1,{H*W,H*W});

    return D+A;
}


int main() {
    timeval time1,time2;

    torch::jit::script::Module tensors = torch::jit::load("../images_jit.pth");
    torch::Tensor fixed = tensors.attr("fixed").toTensor();
    torch::Tensor moving = tensors.attr("moving").toTensor();
    int H = fixed.size(2); int W = fixed.size(3);



    std::cout << "size (fixed) " << H << "x" << W <<"\n";

    //todo: initialize some variables
    auto xy = torch::arange(H * W);
    auto indices = torch::stack({xy, xy}, 0);

    auto L = laplaceMatrix(H,W);
    auto warped = moving;
    auto uv = torch::zeros({1,H,W,2});
    //todo: Print out Name and student-nr
    std::cout << "Solution by: Falco Lentzsch(685454),Konrad von Kuegelgen, Jesse Kruse(675710) \n";
    gettimeofday(&time1, NULL);

    //Task1:



    //todo: calculate the two-point stencil
    auto weight = torch::linspace(-.5, .5, 3);
    auto Mx_weights = weight.view({1, 1, 1, 3});
    auto My_weights = weight.view({1, 1, 3, 1});

    //todo: create vectors u,v


    //todo: Apply the gradient filter using 2d-convolution

    // Input Bilder mit Padding versehen
    auto Mx_pad_input = F::pad(moving,F::PadFuncOptions({1, 1, 0, 0}).mode(torch::kReplicate));
    auto My_pad_input = F::pad(moving,F::PadFuncOptions({0, 0, 1, 1}).mode(torch::kReplicate));
    // Bild filtern in x und y Richtung liefert Gradientenfeld
    auto Mx_gradient = F::conv2d(Mx_pad_input,Mx_weights);
    auto My_gradient = F::conv2d(My_pad_input,My_weights);



    //todo: Create Sparse Matrices containing pointwise squared gradients
    // Gradientenfeld als Vektor und quadriert
    auto Mx_gradient_diag_squared = torch::pow(Mx_gradient.view(-1), 2);
    auto My_gradient_diag_squared = torch::pow(My_gradient.view(-1), 2);
    // Eintragen der quadrierten Gradienten auf der Diagonalen unserer Sparse Matrix

    auto Mxx = torch::sparse_coo_tensor(indices, Mx_gradient_diag_squared, {H * W, H * W});
    auto Myy = torch::sparse_coo_tensor(indices, My_gradient_diag_squared, {H * W, H * W});


    //Maybe check sizes:
    //std::cout << Mx.sizes() << endl;
    //todo: Create tensors bx,by
    auto bx = -(moving-fixed) * Mx_gradient;
    auto by = -(moving-fixed) * My_gradient;

    //todo: Solve u,v and stack accordingly to uv
    auto u = sparseCG(L + Mxx, bx, 25).view({H, W});
    auto v = sparseCG(L + Myy, by, 25).view({H, W});

    // generate identity grid using F.affine_grid
    // Erstellen hier ein Grid mit 6 Freiheitsgraden wobei wir es
    // Standartmäßig so initialisieren, das das grid nichts verändert
    auto identity_grid = F::affine_grid(torch::eye(2, 3).unsqueeze(0), {1, 1, H, W}, true);
    // stacked and reshaped uv im Bereich zwischen -1 und 1
    uv = torch::stack({u / (.5 * (H - 1.0)), v / (.5 * (W - 1.0))}, 2).unsqueeze(0);

    // Addieren von UV udn Grid liefert ein Transformationsfeld
    auto mapped_uv_field = identity_grid + uv;
    warped = F::grid_sample(moving, mapped_uv_field, F::GridSampleFuncOptions().align_corners(true));

    F::

    //Task2:



    //For-loop for iterative warping
    for(int i=0;i<10;i++) {

        //todo: Create Mx,My by applying the gradient filter using 2d-convolution
        // Input Bilder mit Padding versehen
        Mx_pad_input = F::pad(warped,F::PadFuncOptions({1, 1, 0, 0}).mode(torch::kReplicate));
        My_pad_input = F::pad(warped,F::PadFuncOptions({0, 0, 1, 1}).mode(torch::kReplicate));
        // Bild filtern in x und y Richtung liefert Gradientenfeld
        Mx_gradient = F::conv2d(Mx_pad_input,Mx_weights);
        My_gradient = F::conv2d(My_pad_input,My_weights);

        //todo: Create Mxx,Myy
        // Gradientenfeld als Vektor und quadriert
        Mx_gradient_diag_squared = torch::pow(Mx_gradient.view(-1), 2);
        My_gradient_diag_squared = torch::pow(My_gradient.view(-1), 2);
        // Eintragen der quadrierten Gradienten auf der Diagonalen unserer Sparse Matrix

        Mxx = torch::sparse_coo_tensor(indices, Mx_gradient_diag_squared, {H * W, H * W});
        Myy = torch::sparse_coo_tensor(indices, My_gradient_diag_squared, {H * W, H * W});

        //todo: Create bx,by
        bx = - torch::_sparse_mm(L, u.view({-1, 1})).view({H, W}) - (warped-fixed) * Mx_gradient;
        by = - torch::_sparse_mm(L, v.view({-1, 1})).view({H, W}) - (warped-fixed) * My_gradient;

        //todo: Solve u,v
        u += sparseCG(L + Mxx, bx, 25).view({H, W});
        v += sparseCG(L + Myy, by, 25).view({H, W});

        //todo: Scale and stack displacements accordingly
        uv = torch::stack({u / (.5 * (H - 1.0)), v / (.5 * (W - 1.0))}, 2).unsqueeze(0);

        //todo: warp Image using F:grid_sample with identity grid
        mapped_uv_field = identity_grid + uv;
        warped = F::grid_sample(moving, mapped_uv_field, F::GridSampleFuncOptions().align_corners(true));

    }






    //given: Output
    auto output = torch::cat({uv,warped.permute({0,2,3,1})},3);

    gettimeofday(&time2, NULL);

    float timeP=1.0f*(time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6));

    std::cout << "runtime "<<timeP<<" secs\n";
    auto bytes = torch::jit::pickle_save(output.view({H,W,3}));
    std::ofstream fout("../output_flow.pth", std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();

}
