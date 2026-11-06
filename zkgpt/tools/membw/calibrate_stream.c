#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
int main(){ size_t n=1UL<<29; /* 4 GiB doubles each */ double*a=malloc(n*8),*b=malloc(n*8);
 #pragma omp parallel for
 for(size_t i=0;i<n;i++){a[i]=1;b[i]=2;}
 double t=omp_get_wtime(); double s=0;
 for(int r=0;r<5;r++){
 #pragma omp parallel for reduction(+:s)
 for(size_t i=0;i<n;i++) s+=a[i]+b[i]; }
 t=omp_get_wtime()-t; printf("read %.1f GB in %.2fs = %.1f GB/s (s=%g)\n",5*2*n*8/1e9,t,5*2*n*8/1e9/t,s);}
