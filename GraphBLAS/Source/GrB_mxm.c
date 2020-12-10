#include "GB_mxm.h"

extern GrB_Info GB_mxm_gpu 
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
);

GrB_Info GrB_mxm
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' and '*' for T=A*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B,
)
{
    fprintf (stderr, "mxm running\n") ;
    GB_WHERE ("GrB_mxm (C, M, accum, semiring, A, B, desc)") ;
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
                       A_transpose, B_transpose, AxB_method);
    info = GB_mxm_gpu (C, C_replace,
                       M, Mask_comp, Mask_struct,
                       accum,
                       semiring,
                       A, A_transpose,
                       B, B_transpose,
                       false,
                       AxB_method,
                       Context) ;
    printf("Calling ended\n");
    return (info);
}
