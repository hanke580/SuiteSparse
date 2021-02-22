#include "utils.h"

/*
   converts CSR format to CSC format, not in-place,
   if csrVal == NULL, only pattern is reorganized.
   the size of matrix is n x m.
 */

void csr2csc(int n, int m, int nz, int *csrRowPtr, int *csrColIdx, int *csrVal, 
             int *cscColPtr, int *cscRowIdx, int *cscVal)
{
  int i, j, k, l;
  int *ptr;

  for (i=0; i<=m; i++) cscColPtr[i] = 0;

  /* determine column lengths */
  for (i=0; i<nz; i++) cscColPtr[csrColIdx[i]+1]++;

  for (i=0; i<m; i++) cscColPtr[i+1] += cscColPtr[i];


  /* go through the structure once more. Fill in output matrix. */

  for (i=0, ptr=csrRowPtr; i<n; i++, ptr++)
    for (j=*ptr; j<*(ptr+1); j++){
      k = csrColIdx[j];
      l = cscColPtr[k]++;
      cscRowIdx[l] = i;
      if (csrVal) cscVal[l] = csrVal[j];
    }

  /* shift back cscColPtr */
  for (i=m; i>0; i--) cscColPtr[i] = cscColPtr[i-1];

  cscColPtr[0] = 0;
}
