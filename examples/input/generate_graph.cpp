#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using namespace std;

int main() {
    const int N = 10000;
    const int maxOutLinks = 10;
    const double danglingProb = 0.05;

    string fname = "";
    fname << "graph" << N << ".txt";
    ofstream fout(fname);
    mt19937 rng(42);
    uniform_int_distribution<int> outDist(1, maxOutLinks);
    uniform_int_distribution<int> nodeDist(0, N-1);
    uniform_real_distribution<double> prob(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        fout << i;
        if (prob(rng) > danglingProb) {
            int outLinks = outDist(rng);
            for (int j = 0; j < outLinks; ++j) {
                int target = nodeDist(rng);
                fout << " " << target;
            }
        }
        fout << "\n";
    }

    fout.close();
}
