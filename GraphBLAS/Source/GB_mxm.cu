#include "Kernels/spgemm.h"
#include "../Include/GraphBLAS.h"
#include "GB.h"


typedef int TYPE;

extern "C" GrB_Info GB_transpose           // C=A', C=(ctype)A or C=op(A')
(
    GrB_Matrix *Chandle,            // output matrix C, possibly modified in place
    GrB_Type ctype,                 // desired type of C; if NULL use A->type.
                                    // ignored if op is present (cast to op->ztype)
    const bool C_is_csc,            // desired CSR/CSC format of C
    const GrB_Matrix A_in,          // input matrix
    // no operator is applied if both op1 and op2 are NULL
    const GrB_UnaryOp op1_in,       // unary operator to apply
    const GrB_BinaryOp op2_in,      // binary operator to apply
    const GxB_Scalar scalar,        // scalar to bind to binary operator
    bool binop_bind1st,             // if true, binop(x,A) else binop(A,y)
    GB_Context Context
 );

extern "C" GrB_Info GB_mxm_gpu
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // if true, clear C before writing to it
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' and '*' for C=A*B
    const GrB_Matrix A,             // input matrix
    const bool A_transpose,         // if true, use A' instead of A
    const GrB_Matrix B,             // input matrix
    const bool B_transpose,         // if true, use B' instead of B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    const GrB_Desc_Value AxB_method,// for auto vs user selection of methods
    GB_Context Context
)

{
    printf("GPU Calling Starting ---------------------------------------- \n");
    GrB_Info info ;
    if (A_transpose) {
        GB_transpose(NULL, NULL, false, A, NULL, NULL, NULL, false, Context); // 是否inplace
    } else if (A->is_csc == true) {
        printf("A csc not implemented yet__------------------------------\n");
        return GrB_NO_VALUE;
        // csc => csr 参考 csc2csr的实现
    }
    if (B_transpose) {
	GB_transpose(NULL, NULL, true, B, NULL, NULL, NULL, false, Context); // 是否inplace
    } else if (B->is_csc == false) {
        printf("B csr not implemented yet--------------------------------\n");
        return GrB_NO_VALUE;
        // csr => csc
    }

    // Check type A and B, and broadcast. For simplicity, choose A type currently.
    printf("M->nzmax = %ld\n", M->nzmax);
    printf("M->plen  = %ld\n", M->plen);
    
    printf("A->nzmax = %ld\n", A->nzmax);
    printf("A->plen  = %ld\n", A->plen);

    printf("B->plen  = %ld\n", B->plen);
    printf("B->nzmax = %ld\n", B->nzmax);

    printf("C->plen  = %ld\n", C->plen);
    printf("C->nzmax = %ld\n", C->nzmax);
    printf("C->vlen = %ld\n", C->vlen);
    printf("C->vdim = %ld\n", C->vdim);

    TYPE* C_csrVal;
    cudaMalloc(&C_csrVal, M->nzmax * sizeof(TYPE));

    int* A_csrRowPtr;   // GPU CSR format
    int* A_csrColInd;
    TYPE* A_csrVal;        // TODO: Need Correct A TYPE
    cudaMalloc(&A_csrRowPtr, (A->plen + 1) * sizeof(int));
    cudaMalloc(&A_csrColInd, A->nzmax * sizeof(int));
    cudaMalloc(&A_csrVal, A->nzmax * sizeof(TYPE));

    const int A_nrows = A->plen;

    // Alloc space in GPU and transfer memory from CPU to GPU
    cudaMemcpy(A_csrRowPtr, A->p, (A->plen + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_csrColInd, A->i, A->nzmax * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_csrVal,    A->x, A->nzmax * sizeof (TYPE), cudaMemcpyHostToDevice);

    int* B_cscColPtr;   // GPU CSR format
    int* B_cscRowInd;
    TYPE* B_cscVal;        // TODO: Need Correct A TYPE
    cudaMalloc(&B_cscColPtr, (B->plen + 1) * sizeof(int));
    cudaMalloc(&B_cscRowInd, B->nzmax * sizeof(int));
    cudaMalloc(&B_cscVal, B->nzmax * sizeof(TYPE));

    // Alloc space in GPU and transfer memory from CPU to GPU
    cudaMemcpy(B_cscColPtr, B->p, (B->plen + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_cscRowInd, B->i, B->nzmax * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_cscVal,    B->x, B->nzmax * sizeof (TYPE), cudaMemcpyHostToDevice);

    int* M_csrRowPtr;       // GPU CSR format
    int* M_csrColInd;
    int* M_csrVal;             // TODO: Need Correct A TYPE

    cudaMalloc(&M_csrRowPtr, (M->plen + 1) * sizeof(int));
    cudaMalloc(&M_csrColInd, M->nzmax * sizeof(int));
    cudaMalloc(&M_csrVal, M->nzmax * sizeof(int));

    cudaMemcpy(M_csrRowPtr, M->p, (M->plen + 1) * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(M_csrColInd, M->i, M->nzmax * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(M_csrVal,    M->x, M->nzmax * sizeof (int), cudaMemcpyHostToDevice);

    
    int ifprint = 1;
    if(ifprint) {
        printf("A->p\n");
        for(int i = 0; i<A->plen+1; i++) {
            printf("%ld\t", A->p[i]);
        }
        printf("\n");
        printf("A->i\n");
        for(int i = 0; i<A->nzmax; i++) {
            printf("%ld\t", A->i[i]);
        }
        printf("\n");
        printf("A->x\n");
    
        int *Ax = static_cast<int*>(A->x);
        for(int i = 0; i<A->nzmax; i++) {
            printf("%d\t", Ax[i]);
        }
        printf("\n");
    
        printf("B->p\n");
        for(int i = 0; i<B->plen+1; i++) {
            printf("%ld\t", B->p[i]);
        }
        printf("\n");
    
        printf("B->i\n");
        for(int i = 0; i<B->nzmax; i++) {
            printf("%ld\t", B->i[i]);
        }
        printf("\n");
        printf("B->x\n");
    
        int *Bx = static_cast<int*>(B->x);
        for(int i = 0; i<B->nzmax; i++) {
            printf("%d\t", Bx[i]);
        }
        printf("\n");
        printf("M->p\n");
        for(int i = 0; i<M->plen+1; i++) {
            printf("%ld\t", M->p[i]);
        }
        printf("\n");
    
        printf("M->i\n");
        for(int i = 0; i<M->nzmax; i++) {
            printf("%ld\t", M->i[i]);
        }
        printf("\n");
        printf("M->x\n");
    
        int *Mx = static_cast<int*>(M->x);
        for(int i = 0; i<M->nzmax; i++) {
            printf("%d\t", Mx[i]);
        }
        printf("\n");
    
    }
    const int nt = 128;  // GrB_128
    if (M != NULL) {
        printf("Calling Kernel Function\n");
        // Simple warp-per-row algorithm
        dim3 NT(nt), NB(ceil(A_nrows/nt));

        spgemmMaskedKernel<<<NB, NT>>>(C_csrVal,
            M_csrRowPtr,
            M_csrColInd,
            M_csrVal,
            semiring->multiply->function, semiring->add->op->function,
            0,
            A_csrRowPtr, A_csrColInd, A_csrVal,
            B_cscColPtr, B_cscRowInd, B_cscVal,
            A_nrows);

	if(C->x == NULL)
        C->x = (int*)malloc(M->nzmax * sizeof(int));

	int* temp = (int*)malloc(M->nzmax * sizeof(int));
        cudaMemcpy(temp, C_csrVal, M->nzmax * sizeof(TYPE), cudaMemcpyDeviceToHost);

	for(int k = 0; k<M->nzmax; k++){
		printf("%d\t", temp[k]);
	}
    printf("\n\n");
    C->p = M->p;
    C->i = M->i;
    C->x = (void *) temp;
    C->nzmax = M->nzmax;
    }
    else {
        printf("No Mask computing not complement\n");
    }
    printf("GPU Calling end\n");
    return (info) ;
}
