#include<stdio.h>

void csr2csc(int n, int m, int nz, int *csrRowPtr, int *csrColIdx, int *csrVal, 
             int *cscColPtr, int *cscRowIdx, int *cscVal);