#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace torch::indexing;
namespace F = torch::nn::functional;

/**
 *
 * @param my_displacements
 * @param gt_displacements
 * @return target registration error between my and gt
 */
float target_registration_error(torch::Tensor my_displacements, torch::Tensor gt_displacements){
    torch::Tensor norm= torch::tensor({223/2,159/2,223/2},{torch::kFloat64}).view({-1,3});
    torch::Tensor diff = my_displacements*norm - gt_displacements*norm;
    return torch::mean(torch::sqrt(torch::sum(diff.pow(2),1))).item<float>();
}

/**
 *
 * @param kpts_fixed
 * @param kpts_moving
 * @param kC
 * @param D
 * @return candidates for given input kpts_fixed and kpts_moving
 */
torch::Tensor find_candidates(torch::Tensor kpts_fixed, torch::Tensor kpts_moving, int kC, int D){
    // bauen M (kpt_fixed) x N (kpts_moving) matrix mit Pairwise Distance
    torch::Tensor f_m_dist = torch::cdist(kpts_fixed,kpts_moving);
    // Index der jeweils 256 kandidaten für die 2048 bestimmen
    torch::Tensor candidates_idx = std::get<1>(torch::topk(f_m_dist, kC, 1,false,true));
    torch::Tensor candidate_values = torch::gather(kpts_moving.view({-1,1,3}).repeat({1,kC,1}),0,candidates_idx.view({-1,kC,1}).repeat({1,1,D}));
    candidate_values = candidate_values - kpts_fixed.unsqueeze(1); // --> Realtives Displacment Berechnen
    return candidate_values;
}

/**
 *
 * @param kpts_fixed
 * @param E
 * @param P
 * @return return a edge list which is interpreted as a undirected graph
 */
torch::Tensor build_graph(torch::Tensor kpts_fixed, int E, int P) {
    // Hier alle Pairwise Distanzen bestimmen
    torch::Tensor f_f_dis = torch::cdist(kpts_fixed, kpts_fixed);

    // Top 10 Werte nehmen und uns den 10ten anschauen als Schwellwert
    torch::Tensor knn_value = std::get<0>(torch::topk(f_f_dis, E + 1, 1, false, true));
    torch::Tensor knn_highest_value = knn_value.index({Slice(), Slice(E, E + 1)});

    // Filtern der Distanzmatrix  pro Zeile nach Werten, die kleiner sind als der highest_value
    // IDEE_1 um auf 19020 zu kommen --> Threshold für die die Differenz
    // knn_mask = (f_f_dis < knn_highest_value+0.000896).int() - torch.eye(P,P)

    // IDEE_2 Gegenläufige Kanten mit rein nehmen. Angenommmen Kante 0 --> 1  aber keine Kante 1 --> 0,
    // da 1 andere 9 nächste Nachbarn haben kann. Dann nehmen wir die Kante 1 --> 0 trotzdem mit auf,
    // weil wir ja keinen gerichteten Graphen haben und die kante sogesehen eh existiert.
    torch::Tensor knn_mask = (f_f_dis < knn_highest_value).to(torch::kInt64);
    torch::Tensor knn_mask_regular = knn_mask - torch::eye(P, P);
    torch::Tensor knn_mask_tranposed = knn_mask.transpose(0, 1) - torch::eye(P, P);
    knn_mask = knn_mask_regular + knn_mask_tranposed;

    return torch::nonzero(knn_mask);
}



int main() {
    timeval time1, time2;
    //load data
    torch::jit::script::Module tensors = torch::jit::load("../fvmbv_sparse_keypoints_jit.pth");
    torch::Tensor kpts_fixed = tensors.attr("kpts_fixed").toTensor();
    torch::Tensor kpts_moving = tensors.attr("kpts_moving").toTensor();
    torch::Tensor gt_displacements = tensors.attr("gt_displacements").toTensor();

    // Im Folgeden benötigte Variablen initialisieren.
    // P = Points, D = Dimensions, kC = Amount of Candidates, E = Number of Edges for each Point
    int kC = 256;
    int E = 9;
    int P = kpts_fixed.sizes()[0];
    int D = kpts_fixed.sizes()[1];
    int epochs = 7;

    //todo: provide names and student nr_s
    std::cout << "Solution by: Falco Lentzsch\n";

    //your code here
    std::cout <<"ktps_fixed:"<< kpts_fixed.sizes() << std::endl;
    std::cout <<"kpts_moving:"<< kpts_moving.sizes() << std::endl;
    std::cout <<"gt_displacements:"<< gt_displacements.sizes() << std::endl;

    // Aufgabe 1
    torch::Tensor candidates = find_candidates(kpts_fixed, kpts_moving, kC, D);
    std::cout << "cindates shape:" << candidates.sizes() << std::endl;
    torch::Tensor edges = build_graph(kpts_fixed,E,P);
    std::cout << "edges shape:" << edges.sizes() << std::endl;

    // Aufgabe 2
    // Speichert die Nachrichten pro Knoten
    torch::Tensor tmp_msg = torch::zeros({P, kC}); // --> shape (2048x256)
    // Nachrichten für jede Kante. Jede Nachricht hat 256 Kandidaten (0...255) an jeder Kante
    torch::Tensor msg = torch::zeros({edges.sizes()[0], kC}); // --> shape (#edgesx256)
    // Speichert später die gesammelten Nachrichten aller Kanten
    torch::Tensor passed_message = torch::zeros({edges.sizes()[0], kC}); // --> shape (#edgesx256)

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Betrachten hier jede Edge einzeln und berechnen den Message Term, da nicht parallelisiert.
        for (int edge_idx = 0; edge_idx < edges.sizes()[0]; edge_idx++) {
            // Kante von p nach q
            int p = edges[edge_idx][0].item().to<int>();
            int q = edges[edge_idx][1].item().to<int>();

            // Zugehörige Kandidaten von p und q
            torch::Tensor u_p = candidates[p];
            torch::Tensor u_q = candidates[q];

            // Regularisierungsterm bauen als Punktweise Distanzen der Candidaten
            torch::Tensor R = (u_p.unsqueeze(0) - u_q.unsqueeze(1)).pow(2).sum(2);

            // Berechnung der neuen Nachricht für jede Kante
            // Formel: Summe der Nachrichten bis zu Knoten P + 1.5 fachen Regularisierungsterm
            msg[edge_idx] = std::get<0>(torch::min(passed_message[edge_idx].unsqueeze(0) + 1.5 * R, 1));
        }
        // Da wir jetzt alle Nachrichten entlang des Graphen berechnet haben, müssen wir nun unser tmp_msg updaten
        // Wir geben hier also die Nachrichten an die Nachbarknoten weiter.
        tmp_msg.zero_();
        tmp_msg.scatter_add_(0, edges.index({Slice(), Slice(1, 2)}).view({-1, 1}).repeat({1, kC}), msg);

        // Die Nachrichten in jedem Knoten werden hier noch durch den Graph gereicht, damit wir später
        // an jeder Kante durch zugriff auf den gleichen index in passed_message, die Summe erhalten.
        passed_message = torch::gather(tmp_msg, 0, edges.index({Slice(), Slice(0, 1)}).view({-1, 1}).repeat({1, kC}));

        //compute error in mm
        torch::Tensor my_displacements = torch::randn({2048, 3});
        //berechnung des aktuellen Displacements
        my_displacements = (torch::softmax(-50 * tmp_msg, 1).unsqueeze(-1) * candidates).sum(1);

        float error_mm = target_registration_error(my_displacements, gt_displacements);

        gettimeofday(&time2, NULL);
        float timeP = 1.0f * (time2.tv_sec + time2.tv_usec / 1e6 - (time1.tv_sec + time1.tv_usec / 1e6));
        std::cout << "runtime " << timeP << " secs\n";
        std::cout << "Error in mm: " << error_mm << std::endl;

    }

}
