#include <stdio.h>
#include "../../Include/GraphBLAS.h"
#include "utils.h"

// Sparse matrix-Sparse matrix multiplication with sparse matrix mask
// Strategy:
// 1) Loop through mask using 1 warp/row
// 2) For each nonzero (row, col) of mask:
//    i)   initialize each thread to identity
//    ii)  compute dot-product A(row, :) x B(:, col)
//    iii) use warp on each nonzero at float time
//    iv)  tally up accumulated sum using warp reduction
//    v)   write to global memory C_csrVal
__global__ void spgemmMaskedKernel(int*           C_csrVal,
                                   const int* mask_csrRowPtr,
                                   const int* mask_csrColInd,
                                   int*           mask_csrVal,
                                   GxB_binary_function        mul_op,
                                   GxB_binary_function        add_op,
                                   int            identity,
                                   const int* A_csrRowPtr,
                                   const int* A_csrColInd,
                                   const int*     A_csrVal,
                                   const int* B_cscColPtr,
                                   const int* B_cscRowInd,
                                   const int*     B_cscVal,
                                   int        A_nrows) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id;
  printf("thread_id = %d\n",thread_id); 
  if(thread_id % 100 ==0)
  printf("thread_id = %d\n", thread_id);
  

  // int warp_id   = thread_id / 32;
  // int lane_id   = thread_id & (32 - 1);
  // if(warp_id % 50 == 0 && lane_id % 20 == 0)
  //   printf("warp_id = %d, lane_id = %d\n", warp_id, lane_id);
  if (warp_id < A_nrows) {
    int row_start = mask_csrRowPtr[warp_id];
    int row_end   = mask_csrRowPtr[warp_id+1];

    // Entire warp works together on each nonzero
    for (int edge = row_start; edge < row_end; ++edge) {
      // row: warp_id
      // col: mask_csrColInd[edge]
      int target_row = warp_id;
      int target_col = mask_csrColInd[edge];

      float mask_val = mask_csrVal[edge];
      float accumulator = 0;
      if (mask_val) {
        // Load B bounds on which we must do binary search
        int a_start = A_csrRowPtr[target_row];
        int a_end = A_csrRowPtr[target_row + 1];

        int b_start = B_cscColPtr[target_col];
        int b_end = B_cscColPtr[target_col + 1];


        for(int iter = a_start; iter < a_end; iter++) {
          int col = A_csrColInd[iter];
          int B_ind = binarySearch(B_cscRowInd, col, b_start, b_end);
          if(B_ind != -1) {
            float A_t = A_csrVal[iter];
            float B_t = B_cscVal[B_ind];
            float C_t = A_t * B_t;
            accumulator = C_t + accumulator;
          }
        }
        C_csrVal[edge] = accumulator;
      }
    }
  }
}

