#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <numeric>
#include <chrono>

using namespace std;
using namespace std::chrono;

const double DAMPING = 0.85;

void loadInput(const string& filename, unordered_map<string,int>& pageIds, vector<string>& pageNames, vector<vector<int>>& outEdges, vector<vector<int>>& inEdges) {
    ifstream file(filename);
    string line, word;

    auto getId = [&](const string& s) {
        if (!pageIds.count(s)) {
            int idx = pageIds.size();
            pageIds[s] = idx;
            pageNames.push_back(s);
            outEdges.emplace_back();
            inEdges.emplace_back();
        }
        return pageIds[s];
    };

    int u, v;
    while (getline(file, line)) {
        stringstream ss(line);
        ss >> word;
        u = getId(word);

        while (ss >> word) {
            v = getId(word);
            outEdges[u].push_back(v);
            inEdges[v].push_back(u);
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

vector<double> rankPages(unordered_map<string,int>& pageIds, vector<string>& pageNames, vector<vector<int>>& outEdges, vector<vector<int>>& inEdges, int maxSupersteps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    if (rank == 0) {
        n = pageIds.size();
    } 
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int verticesPerProcess = (n + size - 1) / size;
    int verticiesStart = rank * verticesPerProcess;
    int verticiesEnd = min(verticiesStart + verticesPerProcess, n);
    int localN = max(0, verticiesEnd - verticiesStart);

    vector<vector<int>> localOutEdges(localN);
    vector<vector<int>> localInEdges(localN);

    // Distribute graph partitions
    if (rank == 0) {
        for (int process = 1; process < size; ++process) {
            int processStart = process * verticesPerProcess;
            int processEnd   = min(processStart + verticesPerProcess, n);
            int processVertexCount = max(0, processEnd - processStart);

            MPI_Send(&processVertexCount, 1, MPI_INT, process, 0, MPI_COMM_WORLD);

            for (int u = processStart; u < processEnd; ++u) {
                int edgeCount = outEdges[u].size();
                MPI_Send(&edgeCount, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
                MPI_Send(outEdges[u].data(), edgeCount, MPI_INT, process, 0, MPI_COMM_WORLD);

                edgeCount = inEdges[u].size();
                MPI_Send(&edgeCount, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
                MPI_Send(inEdges[u].data(), edgeCount, MPI_INT, process, 0, MPI_COMM_WORLD);
            }
        }

        for (int i = 0; i < localN; ++i) {
            localOutEdges[i] = outEdges[verticiesStart + i];
            localInEdges[i]  = inEdges[verticiesStart + i];
        }
    } else {
        MPI_Recv(&localN, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localOutEdges.resize(localN);
        localInEdges.resize(localN);

        for (int i = 0; i < localN; ++i) {
            int edgeCount;
            MPI_Recv(&edgeCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localOutEdges[i].resize(edgeCount);
            MPI_Recv(localOutEdges[i].data(), edgeCount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(&edgeCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localInEdges[i].resize(edgeCount);
            MPI_Recv(localInEdges[i].data(), edgeCount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // PageRank algorithm
    vector<double> localPageRanks(localN, 1.0 / n);
    vector<double> nextLocalPageRanks(localN, 0.0);
    vector<double> messages(n, 0.0);

    bool messagesSent = true;

    for (int step = 0; step < maxSupersteps && messagesSent; ++step) {
        messagesSent = false;
        fill(nextLocalPageRanks.begin(), nextLocalPageRanks.end(), 0.0);
        fill(messages.begin(), messages.end(), 0.0);

        double localDangling = 0.0;

        for (int i = 0; i < localN; ++i) {
            if (localOutEdges[i].empty()) {
                localDangling += localPageRanks[i];
            } else {
                double share = localPageRanks[i] / localOutEdges[i].size();
                for (int u : localOutEdges[i]) {
                    messages[u] += share;
                } 
                messagesSent = true;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, messages.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double danglingMass = 0.0;
        MPI_Allreduce(&localDangling, &danglingMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double danglingShare = DAMPING * danglingMass / n;

        for (int i = 0; i < localN; ++i) {
            int v = verticiesStart + i;
            nextLocalPageRanks[i] = (1.0 - DAMPING)/n + DAMPING * messages[v] + danglingShare;
        }

        localPageRanks.swap(nextLocalPageRanks);

        int anyMessage = messagesSent ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &anyMessage, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        messagesSent = anyMessage;
    }

    vector<double> pageRanks(n, 0.0);
    vector<int> counts(size), displacements(size);
    for (int i = 0; i < size; ++i) {
        int start = i * verticesPerProcess;
        int end = min(start + verticesPerProcess, n);
        counts[i] = end - start;
        displacements[i] = start;
    }

    MPI_Gatherv(localPageRanks.data(), localN, MPI_DOUBLE, pageRanks.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return pageRanks;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if(rank == 0) {
            cout << "MAX_SUPERSTEPS is missing..." << endl << "Usage: " << argv[0] << " <MAX_SUPERSTEPS>" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    int maxSupersteps = atoi(argv[1]);

    unordered_map<string,int> pageIds;
    vector<string> pageNames;
    vector<vector<int>> outEdges;
    vector<vector<int>> inEdges;

    if (rank == 0) {
        string inputFile = "/app/input/graph.txt";
        loadInput(inputFile, pageIds, pageNames, outEdges, inEdges);
    }

    auto start = high_resolution_clock::now();
    vector<double> pageRanks = rankPages(pageIds, pageNames, outEdges, inEdges, maxSupersteps);
    auto end = high_resolution_clock::now();
    long long executionTime = duration_cast<milliseconds>(end - start).count();

    if (rank == 0) {
        string outputFile = "/app/output/distributed_" + to_string(maxSupersteps) + ".txt";
        generateOutput(outputFile, pageRanks, pageNames, executionTime);
    }

    MPI_Finalize();
    return 0;
}


