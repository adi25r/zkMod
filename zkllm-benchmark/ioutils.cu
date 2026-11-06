#include "ioutils.cuh"
#include <cstdio>
#include <cstdlib>

static FILE* open_or_die(const string& filename, const char* mode)
{
    FILE* file = fopen(filename.c_str(), mode);
    if (!file) {
        perror(("cannot open " + filename).c_str());
        exit(1);
    }
    return file;
}

void savebin(const string& filename, const void* gpudata, uint size)
{
    void* data = malloc(size);
    cudaMemcpy(data, gpudata, size, cudaMemcpyDeviceToHost);

    FILE* file = open_or_die(filename, "wb");
    fwrite(data, 1, size, file);
    fclose(file);
    
    free(data);

}

uint findsize(const string& filename)
{
    FILE* file = open_or_die(filename, "rb");
    fseek(file, 0, SEEK_END);
    uint size = ftell(file);
    fclose(file);
    
    return size;
}

void loadbin(const string& filename, void* gpudata, uint size)
{
    void* data = malloc(size);

    FILE* file = open_or_die(filename, "rb");
    fread(data, 1, size, file);
    fclose(file);
    
    cudaMemcpy(gpudata, data, size, cudaMemcpyHostToDevice);

    free(data);

}