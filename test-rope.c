
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define dim 8 // 64

uint8_t head_size = 8;

int main(void) {
    if (1) {
    for (uint8_t pos=0; pos < 4; pos++) {
        float val2 = pos;
        for (uint8_t i = 0; i < dim; i += 2)
        {
            int head_dim = i % head_size;
            if (head_dim==0) { val2 = pos; };
            float val = pos;
            float hh = head_dim / (float)head_size;
            float p = pow(10000.0, hh);
            val *= 1.0 / pow(10000.0, head_dim / (float)head_size);
            printf("%d:%d:HEAD_SIZE=%d,HEAD_DIM=%d,VAL=%f,VAL2=%f\n",pos,i,head_size,head_dim,val,val2);
            printf("\tHH=%f,P=%f\n",hh,p);
            val2 /= 10;
        }
    }
    }
    float x;

    x = pow(10000.0, 0);
    printf("X=%f\t",x);
    x*=2.0;
    printf("X=%f\tEXPECTED 2.0\n",x);

    x = pow(10000.0, 0);
    printf("X=%f\t",x);
    x=2.0/x;
    printf("X=%f\tEXPECTED 2.0\n",x);

    x = 1.0 / pow(10000.0, 0);
    printf("X=%f\t",x);
    x *= 2.0;
    printf("X=%f\tEXPECTED 2.0\n",x);

    x = 2.0;
    printf("X=%f\t",x);
    x *= 1.0 / pow(10000.0, 0);
    printf("X=%f\tEXPECTED 2.0\n",x);
}
