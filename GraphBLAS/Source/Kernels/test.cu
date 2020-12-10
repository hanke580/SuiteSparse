#include <stdio.h>
#include "spgemm.h"

int main() {
    int ap_[] = {0,0,1,2,3};
    int ai_[] = {0,1,2};
    int ax_[] = {1,1,1};

    int bp_[] = {0,0,1,2,3};
    int bi_[] = {0,1,2};
    int bx_[] = {1,1,1};

    int mp_[] = {0,0,1,2,3};
    int mi_[] = {0,1,2};
    int mx_[] = {1,1,1};

    int *ap, *ai, *ax, *bp, *bi, *bx,* mp, *mi, *mx, *cp, *ci, *cx;

    cudaMallocManaged((void**)&ap, 5 * sizeof(int));
    cudaMallocManaged((void**)&ai, 3 * sizeof(int));
    cudaMallocManaged((void**)&ax, 3 * sizeof(int));

    cudaMallocManaged((void**)&bp, 5 * sizeof(int));
    cudaMallocManaged((void**)&bi, 3 * sizeof(int));
    cudaMallocManaged((void**)&bx, 3 * sizeof(int));

    cudaMallocManaged((void**)&mp, 5 * sizeof(int));
    cudaMallocManaged((void**)&mi, 3 * sizeof(int));
    cudaMallocManaged((void**)&mx, 3 * sizeof(int));

    cudaMallocManaged((void**)&cp, 5 * sizeof(int));
    cudaMallocManaged((void**)&ci, 3 * sizeof(int));
    cudaMallocManaged((void**)&cx, 3 * sizeof(int));

    memcpy(ap, ap_, 3 * sizeof(int));
    memcpy(ai, ai_, 5 * sizeof(int));
    memcpy(ax, ax_, 5 * sizeof(int));

    memcpy(bp, bp_, 4 * sizeof(int));
    memcpy(bi, bi_, 4 * sizeof(int));
    memcpy(bx, bx_, 4 * sizeof(int));

    memcpy(mp, mp_, 3 * sizeof(int));
    memcpy(mi, mi_, 6 * sizeof(int));
    memcpy(mx, mx_, 6 * sizeof(int));

    int A_nrows = 2;

    const int nt = 128;
    dim3 NT(nt);
    dim3 NB(1);

    spgemmMaskedKernel<<<NB, NT>>> (
        cx, mp, mi, mx,
        NULL, NULL,
        0,
        ap, ai, ax,
        bp, bi, bx,
        A_nrows
    );
    cudaDeviceSynchronize();
    
    printf("cx value: \n");
    for(int i = 0; i<6; i++){
        printf("%d\t", cx[i]);
    }
    printf("\n");
}
