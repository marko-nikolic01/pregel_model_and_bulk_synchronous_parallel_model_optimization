#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

const double DAMPING = 0.85;

void loadInput(const string& filename, unordered_map<string, int>& pageIds, vector<string>& pageNames, vector<vector<int>>& outEdges) {
    ifstream file(filename);
    string line, word;

    auto getId = [&](const string& s) {
        if (!pageIds.count(s)) {
            int idx = pageIds.size();
            pageIds[s] = idx;
            pageNames.push_back(s);
            outEdges.emplace_back();
        }
        return pageIds[s];
    };

    int v, u;
    while (getline(file, line)) {
        stringstream ss(line);
        ss >> word;
        u = getId(word);

        while (ss >> word) {
            v = getId(word);
            outEdges[u].push_back(v);
        }
    }
}

void generateOutput(const string& filename, const vector<double>& pageRanks, const vector<string>& pageNames, long long executionTime) {
    ofstream outFile(filename);

    outFile << executionTime << endl;
    for (size_t i = 0; i < pageRanks.size(); ++i) {
        outFile << pageNames[i] << " " << pageRanks[i] << endl;
    }

    outFile.close();
}

vector<double> rankPages(unordered_map<string, int>& pageIds, vector<string>& pageNames, vector<vector<int>>& outEdges, int maxSupersteps) {
    int n = pageIds.size();

    vector<double> pageRanks(n, 1.0 / n);
    vector<double> nextPageRanks(n, 0.0);
    vector<double> inbox(n, 0.0);
    vector<double> outbox(n, 0.0);

    double danglingMass;
    bool messagesSent = true;

    int numThreads = omp_get_max_threads();

    for (int step = 0; step < maxSupersteps && messagesSent; ++step) {
        danglingMass = 0.0;
        messagesSent = false;

        fill(outbox.begin(), outbox.end(), 0.0);

        #pragma omp parallel for reduction(|:messagesSent) reduction(+:danglingMass)
        for (int v = 0; v < n; ++v) {
            double sum = inbox[v];
            nextPageRanks[v] = (1.0 - DAMPING) / n + DAMPING * sum;

            if (outEdges[v].empty()) {
                danglingMass += pageRanks[v];
            } else {
                double share = pageRanks[v] / outEdges[v].size();
                for (int u : outEdges[v]) {
                    #pragma omp atomic
                    outbox[u] += share;
                    messagesSent = true;
                }
            }
        }

        double danglingShare = DAMPING * danglingMass / n;

        #pragma omp parallel for
        for (int v = 0; v < n; ++v) {
            nextPageRanks[v] += danglingShare;
        }

        swap(inbox, outbox);
        pageRanks.swap(nextPageRanks);
        fill(nextPageRanks.begin(), nextPageRanks.end(), 0.0);
    }
    
    return pageRanks;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "MAX_SUPERSTEPS is missing..." << endl
             << "Usage: " << argv[0] << " <MAX_SUPERSTEPS>" << endl;
        return 1;
    }

    int maxSupersteps = atoi(argv[1]);

    unordered_map<string, int> pageIds;
    vector<string> pageNames;
    vector<vector<int>> outEdges;

    string inputFile = "./examples/input/graph.txt";
    loadInput(inputFile, pageIds, pageNames, outEdges);

    auto start = high_resolution_clock::now();
    vector<double> pageRanks = rankPages(pageIds, pageNames, outEdges, maxSupersteps);
    auto end = high_resolution_clock::now();
    long long executionTime =duration_cast<milliseconds>(end - start).count();

    cout << "Execution time: " << executionTime << " ms" << endl;

    string outputFile = "./examples/output/parallel" + to_string(maxSupersteps) + ".txt";
    generateOutput(outputFile, pageRanks, pageNames, executionTime);

    return 0;
}
