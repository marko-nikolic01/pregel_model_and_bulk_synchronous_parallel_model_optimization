#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <CL/cl.h>

using namespace std;
using namespace std::chrono;

const double DAMPING = 0.85;

const char* kernelSource = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline void atomic_add_double(__global double* addr, double val) {
    union {
        unsigned long u64;
        double f64;
    } old_val, new_val;
    
    do {
        old_val.f64 = *addr;
        new_val.f64 = old_val.f64 + val;
    } while (atom_cmpxchg((__global unsigned long*)addr, old_val.u64, new_val.u64) != old_val.u64);
}

__kernel void pageRankKernel(
    __global const double* inbox,
    __global const double* pageRanks,
    __global const int* offsets,
    __global const int* edges,
    __global double* nextPageRanks,
    __global double* outbox,
    int n,
    double damping)
{
    int v = get_global_id(0);
    if (v >= n) return;

    double sum = inbox[v];
    nextPageRanks[v] = (1.0 - damping) / n + damping * sum;

    int start = offsets[v];
    int end = offsets[v + 1];
    
    if (start < end) {
        double share = pageRanks[v] / (end - start);
        for (int i = start; i < end; ++i) {
            int u = edges[i];
            atomic_add_double(&outbox[u], share);
        }
    }
}

__kernel void danglingMassKernel(
    __global const double* pageRanks,
    __global const int* offsets,
    __global double* danglingMass,
    int n)
{
    __local double localSum[256];
    
    int lid = get_local_id(0);
    int v = get_global_id(0);
    
    double sum = 0.0;
    if (v < n && offsets[v] == offsets[v + 1]) {
        sum = pageRanks[v];
    }
    
    localSum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localSum[lid] += localSum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add_double(danglingMass, localSum[0]);
    }
}

__kernel void addDanglingMassKernel(
    __global double* nextPageRanks,
    double danglingShare,
    int n)
{
    int v = get_global_id(0);
    if (v >= n) return;
    nextPageRanks[v] += danglingShare;
}
)";

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        cerr << "Error during operation '" << operation << "': " << err << endl;
        exit(1);
    }
}

void loadInput(const string& filename, unordered_map<string, int>& pageIds, vector<string>& pageNames, vector<int>& edges, vector<int>& offsets) {
    ifstream file(filename);
    string line, word;

    auto getId = [&](const string& s) {
        if (!pageIds.count(s)) {
            int idx = pageIds.size();
            pageIds[s] = idx;
            pageNames.push_back(s);
        }
        return pageIds[s];
    };

    vector<vector<int>> tmpEdges;

    while (getline(file, line)) {
        stringstream ss(line);
        ss >> word;
        int u = getId(word);

        if ((int)tmpEdges.size() <= u) {
            tmpEdges.resize(u + 1);
        }

        while (ss >> word) {
            int v = getId(word);
            tmpEdges[u].push_back(v);
        }
    }

    offsets.push_back(0);
    for (const auto& vec : tmpEdges) {
        edges.insert(edges.end(), vec.begin(), vec.end());
        offsets.push_back(edges.size());
    }

    while ((int)offsets.size() <= (int)pageIds.size()) {
        offsets.push_back(edges.size());
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

vector<double> rankPages(unordered_map<string, int>& pageIds, vector<string>& pageNames, vector<int>& edges, vector<int>& offsets, int maxSupersteps) {
    int n = pageIds.size();
    int m = edges.size();

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "clGetPlatformIDs");
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        checkError(err, "clGetDeviceIDs");
    }
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");
    
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkError(err, "clCreateProgramWithSource");
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        cerr << "Build error:\n" << log << endl;
        delete[] log;
        exit(1);
    }
    
    cl_kernel pageRankKernel = clCreateKernel(program, "pageRankKernel", &err);
    checkError(err, "clCreateKernel pageRankKernel");
    
    cl_kernel danglingMassKernel = clCreateKernel(program, "danglingMassKernel", &err);
    checkError(err, "clCreateKernel danglingMassKernel");
    
    cl_kernel addDanglingMassKernel = clCreateKernel(program, "addDanglingMassKernel", &err);
    checkError(err, "clCreateKernel addDanglingMassKernel");

    vector<double> h_pageRanks(n, 1.0 / n);
    vector<double> h_inbox(n, 0.0);

    cl_mem d_pageRanks = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * sizeof(double), h_pageRanks.data(), &err);
    checkError(err, "clCreateBuffer pageRanks");
    
    cl_mem d_nextPageRanks = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer nextPageRanks");
    
    cl_mem d_inbox = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * sizeof(double), h_inbox.data(), &err);
    checkError(err, "clCreateBuffer inbox");
    
    cl_mem d_outbox = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer outbox");
    
    cl_mem d_edges = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(int), edges.data(), &err);
    checkError(err, "clCreateBuffer edges");
    
    cl_mem d_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (n + 1) * sizeof(int), offsets.data(), &err);
    checkError(err, "clCreateBuffer offsets");
    
    cl_mem d_danglingMass = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer danglingMass");

    size_t globalWorkSize = ((n + 255) / 256) * 256;
    size_t localWorkSize = 256;

    for (int step = 0; step < maxSupersteps; ++step) {
        double zero = 0.0;
        err = clEnqueueFillBuffer(queue, d_outbox, &zero, sizeof(double), 0, n * sizeof(double), 0, NULL, NULL);
        checkError(err, "clEnqueueFillBuffer outbox");
        
        err = clEnqueueFillBuffer(queue, d_danglingMass, &zero, sizeof(double), 0, sizeof(double), 0, NULL, NULL);
        checkError(err, "clEnqueueFillBuffer danglingMass");

        clSetKernelArg(pageRankKernel, 0, sizeof(cl_mem), &d_inbox);
        clSetKernelArg(pageRankKernel, 1, sizeof(cl_mem), &d_pageRanks);
        clSetKernelArg(pageRankKernel, 2, sizeof(cl_mem), &d_offsets);
        clSetKernelArg(pageRankKernel, 3, sizeof(cl_mem), &d_edges);
        clSetKernelArg(pageRankKernel, 4, sizeof(cl_mem), &d_nextPageRanks);
        clSetKernelArg(pageRankKernel, 5, sizeof(cl_mem), &d_outbox);
        clSetKernelArg(pageRankKernel, 6, sizeof(int), &n);
        clSetKernelArg(pageRankKernel, 7, sizeof(double), &DAMPING);

        err = clEnqueueNDRangeKernel(queue, pageRankKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel pageRankKernel");

        clSetKernelArg(danglingMassKernel, 0, sizeof(cl_mem), &d_pageRanks);
        clSetKernelArg(danglingMassKernel, 1, sizeof(cl_mem), &d_offsets);
        clSetKernelArg(danglingMassKernel, 2, sizeof(cl_mem), &d_danglingMass);
        clSetKernelArg(danglingMassKernel, 3, sizeof(int), &n);

        err = clEnqueueNDRangeKernel(queue, danglingMassKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel danglingMassKernel");

        double danglingMass;
        err = clEnqueueReadBuffer(queue, d_danglingMass, CL_TRUE, 0, sizeof(double), &danglingMass, 0, NULL, NULL);
        checkError(err, "clEnqueueReadBuffer danglingMass");

        double danglingShare = DAMPING * danglingMass / n;

        clSetKernelArg(addDanglingMassKernel, 0, sizeof(cl_mem), &d_nextPageRanks);
        clSetKernelArg(addDanglingMassKernel, 1, sizeof(double), &danglingShare);
        clSetKernelArg(addDanglingMassKernel, 2, sizeof(int), &n);

        err = clEnqueueNDRangeKernel(queue, addDanglingMassKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel addDanglingMassKernel");

        err = clEnqueueCopyBuffer(queue, d_outbox, d_inbox, 0, 0, n * sizeof(double), 0, NULL, NULL);
        checkError(err, "clEnqueueCopyBuffer");
        
        swap(d_pageRanks, d_nextPageRanks);
    }

    err = clEnqueueReadBuffer(queue, d_pageRanks, CL_TRUE, 0, n * sizeof(double), h_pageRanks.data(), 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer pageRanks");

    clReleaseMemObject(d_pageRanks);
    clReleaseMemObject(d_nextPageRanks);
    clReleaseMemObject(d_inbox);
    clReleaseMemObject(d_outbox);
    clReleaseMemObject(d_edges);
    clReleaseMemObject(d_offsets);
    clReleaseMemObject(d_danglingMass);
    clReleaseKernel(pageRankKernel);
    clReleaseKernel(danglingMassKernel);
    clReleaseKernel(addDanglingMassKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return h_pageRanks;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "MAX_SUPERSTEPS is missing..." << endl << "Usage: " << argv[0] << " <MAX_SUPERSTEPS>" << endl;
        return 1;
    }
    int maxSupersteps = atoi(argv[1]);

    unordered_map<string, int> pageIds;
    vector<string> pageNames;
    vector<int> edges;
    vector<int> offsets;

    string inputFile = "/app/input/graph.txt";
    loadInput(inputFile, pageIds, pageNames, edges, offsets);

    auto start = high_resolution_clock::now();
    vector<double> pageRanks = rankPages(pageIds, pageNames, edges, offsets, maxSupersteps);
    auto end = high_resolution_clock::now();
    long long executionTime = duration_cast<milliseconds>(end - start).count();

    string outputFile = "/app/output/accelerated_" + to_string(maxSupersteps) + ".txt";
    generateOutput(outputFile, pageRanks, pageNames, executionTime);

    return 0;
}
