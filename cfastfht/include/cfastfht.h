#ifndef CFASTFHT_H
#define CFASTFHT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CFASTFHT_OK 0
#define CFASTFHT_ERR_BAD_INPUT 1
#define CFASTFHT_ERR_ALLOC 2
#define CFASTFHT_ERR_FINUFFT 3
#define CFASTFHT_ERR_INTERNAL 4

typedef struct cfastfht_plan cfastfht_plan;

typedef struct {
    int max_levels;        /* <=0 means auto */
    size_t min_dim_prod;   /* 0 means default */
    double z_split;        /* NaN means auto */
    int K_asy;             /* <0 means auto */
    int K_loc;             /* <0 means auto */
} cfastfht_options;

cfastfht_plan *cfastfht_plan_create(double nu,
                                    const double *rs,
                                    size_t rs_len,
                                    const double *ws,
                                    size_t ws_len,
                                    double tol,
                                    const cfastfht_options *options);

void cfastfht_plan_destroy(cfastfht_plan *plan);

int cfastfht_plan_execute(const cfastfht_plan *plan,
                          const double *cs,
                          double *out);

int cfastfht_plan_execute_batch(const cfastfht_plan *plan,
                                const double *coeffs,
                                size_t coeff_stride,
                                double *out,
                                size_t out_stride,
                                size_t batch_size);

const char *cfastfht_last_error(void);
const char *cfastfht_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif /* CFASTFHT_H */
