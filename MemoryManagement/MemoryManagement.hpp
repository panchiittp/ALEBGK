#ifndef MEMORYMANAGEMENT_HPP
#define MEMORYMANAGEMENT_HPP
// CUDA error checking macro
#define CUDA_CHECK(call)                                                                                    \
    do                                                                                                      \
    {                                                                                                       \
        cudaError_t err = call;                                                                             \
        if (err != cudaSuccess)                                                                             \
        {                                                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(err);                                                                                      \
        }                                                                                                   \
    } while (0)
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getfreememinfo(std::string text)
{
    // auto startalloc = std::chrono::high_resolution_clock::now();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    checkCudaError(err, "cudaMemGetInfo failed");
    double free_mem_gb = static_cast<double>(free_mem) / (1 << 30);
    double total_mem_gb = static_cast<double>(total_mem) / (1 << 30);
    double used_mem_gb = total_mem_gb - free_mem_gb;
    std::cout << "After the Operations : " << text << std::endl;
    std::cout << "Total GPU memory: " << total_mem_gb << " GB" << std::endl;
    std::cout << "Free GPU memory: " << free_mem_gb << " GB" << std::endl;
    std::cout << "Used GPU memory: " << used_mem_gb << " GB" << std::endl;
     
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // auto stopalloc = std::chrono::high_resolution_clock::now();
    // auto GenDurationalloc = std::chrono::duration_cast<std::chrono::seconds>(stopalloc - startalloc);    
    //std::cout << "Time Taken For Device Synchronization "  << count << "is " << GenDurationalloc.count() << " seconds" << std::endl;   
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for Device Synchronization: " << milliseconds/1000.0f << " seconds==========" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}
#endif