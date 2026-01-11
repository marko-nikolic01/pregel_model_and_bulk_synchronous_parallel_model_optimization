# Pregel Model & Bulk Synchronous Parallel Model Optimization

This repository contains C++ implementations of the PageRank algorithm using the Pregel vertex-centric model and the Bulk Synchronous Parallel (BSP) paradigm, including sequential and parallel versions with OpenMP, OpenMPI, and OpenCL, along with performance comparisons and speedup analysis.

</br>
<p align="center">
<img height="175" alt="ISO_C++_Logo svg" src="https://github.com/user-attachments/assets/a9a751ca-603a-48c8-840f-cdaeb46b13e4" />
</p></br>

<p align="center">
<img height="150" alt="OpenMP_logo" src="https://github.com/user-attachments/assets/f0e7a452-dc5b-419c-ba91-6dceea02728b" />
</p></br>

<p align="center">
<img height="250" alt="OpenMPi_logo" src="https://github.com/user-attachments/assets/f006b23e-b1f1-447b-95e5-1d76a0a6c917" />
</p></br>

<p align="center">
<img height="300" alt="images (2)" src="https://github.com/user-attachments/assets/66e153f6-a720-4ee0-880a-154783735312" />
</p></br>


## Results

### Execution time

<p align="center">
  <img width="1920" height="1440" alt="execution_times_linear" src="https://github.com/user-attachments/assets/0123e3c2-33f6-4a99-9977-c7e57ea70867" />
</p>

<p align="center">
<img width="1920" height="1440" alt="execution_times_log" src="https://github.com/user-attachments/assets/acad4451-25da-4689-8dbb-e673a4a2f9c1" />
</p></br>

### Speedup

<p align="center">
<img width="1920" height="1440" alt="speedups_linear" src="https://github.com/user-attachments/assets/f13c4660-bc7f-4869-8ff2-02d7a12e6b4f" />
</p>

<p align="center">
<img width="1920" height="1440" alt="speedups_log" src="https://github.com/user-attachments/assets/d8375f26-811d-49fc-81e4-cbf91049e40e" />
</p></br>

## How to run

### Run container

```
cd examples && \
./run.sh
```

### Stop container

```
cd examples && \
./stop.sh
```

