#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

const double DAMPING = 0.85;

void loadInput(const string& filename, unordered_map<string, int>& id, vector<string>& rid, vector<vector<int>>& outEdges) {
    ifstream file(filename);
    string line, word;

    auto getId = [&](const string& s) {
        if (!id.count(s)) {
            int idx = id.size();
            id[s] = idx;
            rid.push_back(s);
            outEdges.emplace_back();
        }
        return id[s];
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

vector<double> rankPages(const unordered_map<string, int>& pageIds, const vector<string>& pageNames, const vector<vector<int>>& outEdges, int maxSupersteps) {
    int n = pageIds.size();

    vector<double> pageRanks(n, 1.0 / n);
    vector<double> nextPageRanks(n, 0.0);
    vector<vector<double>> inbox(n);
    vector<vector<double>> outbox(n);

    double danglingMass, sum, share, danglingShare;
    bool messagesSent = true;

    for (int step = 0; step < maxSupersteps && messagesSent; ++step) {
        danglingMass = 0.0;
        messagesSent = false;

        for (int v = 0; v < n; ++v) {
            sum = 0.0;
            for (double msg : inbox[v]) {
                sum += msg;
            }

            nextPageRanks[v] = (1.0 - DAMPING) / n + DAMPING * sum;

            if (outEdges[v].empty()) {
                danglingMass += pageRanks[v];
            } else {
                share = pageRanks[v] / outEdges[v].size();
                for (int u : outEdges[v]) {
                    outbox[u].push_back(share);
                    messagesSent = true;
                }
            }
        }

        danglingShare = DAMPING * danglingMass / n;

        for (int v = 0; v < n; ++v) {
            nextPageRanks[v] += danglingShare;
        }

        inbox.swap(outbox);
        for (auto& box : outbox) {
            box.clear();
        }

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

    string outputFile = "./examples/output/sequential" + to_string(maxSupersteps) + ".txt";
    generateOutput(outputFile, pageRanks, pageNames, executionTime);

    return 0;
}
