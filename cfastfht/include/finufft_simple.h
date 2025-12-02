#ifndef CFASTFHT_FINUFFT_SIMPLE_H
#define CFASTFHT_FINUFFT_SIMPLE_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

struct finufft_opts;

int finufft1d3(long long nj,
               const double *x,
               const double complex *c,
               int iflag,
               double eps,
               long long nk,
               const double *s,
               double complex *f,
               const struct finufft_opts *opts);

#ifdef __cplusplus
}
#endif

#endif /* CFASTFHT_FINUFFT_SIMPLE_H */
