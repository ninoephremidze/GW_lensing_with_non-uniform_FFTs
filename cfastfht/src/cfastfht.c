#include "cfastfht.h"
#include "finufft_simple.h"

#include <complex.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

static __thread char g_last_error[512];

static void set_last_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
}

const char *cfastfht_last_error(void) {
    return g_last_error;
}

const char *cfastfht_strerror(int code) {
    switch (code) {
        case CFASTFHT_OK:
            return "ok";
        case CFASTFHT_ERR_BAD_INPUT:
            return "bad input";
        case CFASTFHT_ERR_ALLOC:
            return "allocation failed";
        case CFASTFHT_ERR_FINUFFT:
            return "finufft failure";
        default:
            return "internal error";
    }
}

typedef struct {
    size_t i0; /* 1-based inclusive */
    size_t i1;
    size_t j0;
    size_t j1;
} FHTBox;

typedef struct {
    FHTBox *data;
    size_t size;
    size_t capacity;
} FHTBoxList;

static void boxlist_init(FHTBoxList *list) {
    list->data = NULL;
    list->size = 0;
    list->capacity = 0;
}

static void boxlist_free(FHTBoxList *list) {
    free(list->data);
    list->data = NULL;
    list->size = 0;
    list->capacity = 0;
}

static int boxlist_reserve(FHTBoxList *list, size_t needed) {
    if (needed <= list->capacity) {
        return CFASTFHT_OK;
    }
    size_t new_cap = list->capacity == 0 ? 4 : list->capacity;
    while (new_cap < needed) {
        if (new_cap > SIZE_MAX / 2) {
            return CFASTFHT_ERR_ALLOC;
        }
        new_cap *= 2;
    }
    FHTBox *ptr = realloc(list->data, new_cap * sizeof(FHTBox));
    if (!ptr) {
        return CFASTFHT_ERR_ALLOC;
    }
    list->data = ptr;
    list->capacity = new_cap;
    return CFASTFHT_OK;
}

static int boxlist_push(FHTBoxList *list, FHTBox box) {
    int rc = boxlist_reserve(list, list->size + 1);
    if (rc != CFASTFHT_OK) {
        return rc;
    }
    list->data[list->size++] = box;
    return CFASTFHT_OK;
}

typedef struct cfastfht_plan {
    double nu;
    double tol;
    size_t m;
    size_t n;
    double *rs;
    double *ws;
    double z_split;
    int K_asy;
    int K_loc;
    size_t min_dim_prod;
    int max_levels;
    double *asy_coef;
    size_t asy_coef_len;
    FHTBoxList loc_boxes;
    FHTBoxList asy_boxes;
    FHTBoxList dir_boxes;
    double complex *in_buffer;
    double complex *out_buffer;
    double *real_buffer_1;
    double *real_buffer_2;
    double *cheb_buffer;
    double *bessel_buffer_1;
    double *bessel_buffer_2;
} cfastfht_plan;

static bool is_sorted_strict(const double *arr, size_t len) {
    for (size_t i = 1; i < len; ++i) {
        if (!(arr[i] > arr[i - 1])) {
            return false;
        }
    }
    return true;
}

static double clamp_value(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static double hankel_a(int k, double nu) {
    if (k == 0) {
        return 1.0;
    }
    double prod = 1.0;
    for (int m = 1; m <= k; ++m) {
        double term = 4.0 * nu * nu - (double)(2 * m - 1) * (double)(2 * m - 1);
        prod *= term;
    }
    double denom = tgamma((double)k + 1.0) * pow(8.0, (double)k);
    return prod / denom;
}

static double asy_error_bound(double nu, int K, double z) {
    double term1 = fabs(hankel_a(2 * K, nu)) / pow(z, 2.0 * K);
    double term2 = fabs(hankel_a(2 * K + 1, nu)) / pow(z, 2.0 * K + 1);
    double pref = sqrt(2.0 / (M_PI * z));
    return pref * (term1 + term2);
}

static double find_z_split(double nu, int K, double tol) {
    double lower = 1.0;
    double upper = lower;
    double val = asy_error_bound(nu, K, upper);
    int guard = 0;
    while (val > tol && guard < 60) {
        upper *= 2.0;
        val = asy_error_bound(nu, K, upper);
        guard++;
    }
    if (guard == 60 && val > tol) {
        return NAN;
    }
    lower = upper / 2.0;
    for (int iter = 0; iter < 80; ++iter) {
        double mid = 0.5 * (lower + upper);
        double fmid = asy_error_bound(nu, K, mid);
        if (fmid > tol) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    return upper;
}

static double psi_func(double p) {
    if (p <= 0.0 || p >= 1.0) {
        return 0.0;
    }
    return log(p) + sqrt(1.0 - p * p) - log(1.0 + sqrt(1.0 - p * p));
}

static double wimp_error_bound(double nu, int K, double z) {
    double denom = 2.0 * K + nu;
    if (denom <= 0.0) {
        return 0.0;
    }
    double b_K = psi_func(z / (2.0 * K + 2.0 + nu));
    double term;
    if (2 * K - nu > 0 && z / (2.0 * K - nu) <= 1.0) {
        double c_K = psi_func(z / (2.0 * K + 2.0 - nu));
        term = 2.0 * exp(b_K * (nu / 2.0 + K + 1.0) + c_K * (-nu / 2.0 + K + 1.0)) /
               (1.0 - exp(b_K + c_K));
    } else {
        term = 2.0 * exp(b_K * (nu / 2.0 + K + 1.0)) /
               (1.0 - exp(b_K));
    }
    return term;
}

static int compute_wimp_K(double nu, int K_asy, double z, double tol) {
    int wimp_K = (int)ceil((z - nu) / 2.0);
    int guard = 0;
    while (guard < 2000) {
        double bound = wimp_error_bound(nu, wimp_K, z);
        if (bound < tol) {
            return wimp_K;
        }
        wimp_K += 1;
        guard++;
    }
    return -1;
}

static size_t find_first_gt(const double *arr, size_t len, double value) {
    for (size_t i = 0; i < len; ++i) {
        if (arr[i] > value) {
            return i + 1; /* 1-based */
        }
    }
    return 0;
}

static int split_box(const cfastfht_plan *plan,
                     const FHTBox *box,
                     size_t *out_i,
                     size_t *out_j) {
    const double *rs = plan->rs;
    const double *ws = plan->ws;
    const double z = plan->z_split;
    const size_t num_check = 10;
    size_t width = box->i1 - box->i0 + 1;
    size_t samples = MIN(width, num_check);
    double best_val = -1.0;
    size_t best_i = box->i0;
    size_t last_i = 0;
    for (size_t s = 0; s < samples; ++s) {
        size_t idx;
        if (samples == 1) {
            idx = box->i0;
        } else {
            double t = (double)s / (double)(samples - 1);
            idx = box->i0 + (size_t)llround(t * (double)(width - 1));
        }
        if (idx == last_i) {
            continue;
        }
        last_i = idx;
        double thresh = z / ws[idx - 1];
        size_t j = find_first_gt(rs, plan->m, thresh);
        if (j == 0) {
            j = plan->m;
        }
        if (j <= box->j0) {
            j = box->j0 + 1;
        }
        if (j > box->j1) {
            j = box->j1;
        }
        double val = (double)(idx - box->i0 + 1) * (double)(j - box->j0 + 1) +
                     (double)(box->i1 - idx + 1) * (double)(box->j1 - j + 1);
        if (val > best_val) {
            best_val = val;
            best_i = idx;
            *out_j = j;
        }
    }
    *out_i = best_i;
    return CFASTFHT_OK;
}

static int generate_boxes(const cfastfht_plan *plan,
                          FHTBoxList *loc_boxes,
                          FHTBoxList *asy_boxes,
                          FHTBoxList *dir_boxes) {
    const double *rs = plan->rs;
    const double *ws = plan->ws;
    const size_t n = plan->n;
    const size_t m = plan->m;
    const double z_split = plan->z_split;

    if (plan->K_asy < 0 && plan->K_loc < 0) {
        return boxlist_push(dir_boxes, (FHTBox){1, n, 1, m});
    }

    size_t i0 = find_first_gt(ws, n, z_split / rs[m - 1]);
    size_t i1;
    if (m == 1) {
        i1 = (i0 == 0) ? 0 : i0 - 1;
    } else {
        size_t temp = find_first_gt(ws, n, z_split / rs[0]);
        i1 = (temp == 0) ? n : temp - 1;
    }

    size_t j0 = find_first_gt(rs, m, z_split / ws[n - 1]);
    size_t j1;
    if (n == 1) {
        j1 = (j0 == 0) ? 0 : j0 - 1;
    } else {
        size_t temp = find_first_gt(rs, m, z_split / ws[0]);
        j1 = (temp == 0) ? m : temp - 1;
    }

    if (i0 == 0) {
        return boxlist_push(loc_boxes, (FHTBox){1, n, 1, m});
    }
    if (i1 == 0) {
        return boxlist_push(asy_boxes, (FHTBox){1, n, 1, m});
    }

    if (i0 > 1) {
        int rc = boxlist_push(loc_boxes, (FHTBox){1, i0 - 1, 1, m});
        if (rc != CFASTFHT_OK) return rc;
    }
    if (j0 > 1) {
        int rc = boxlist_push(loc_boxes, (FHTBox){i0, n, 1, j0 - 1});
        if (rc != CFASTFHT_OK) return rc;
    }
    if (i1 < n) {
        int rc = boxlist_push(asy_boxes, (FHTBox){i1 + 1, n, 1, m});
        if (rc != CFASTFHT_OK) return rc;
    }
    if (j1 < m) {
        int rc = boxlist_push(asy_boxes, (FHTBox){1, i1, j1 + 1, m});
        if (rc != CFASTFHT_OK) return rc;
    }

    if (m == 1 || n == 1) {
        return CFASTFHT_OK;
    }

    int rc = boxlist_push(dir_boxes, (FHTBox){i0, i1, j0, j1});
    if (rc != CFASTFHT_OK) {
        return rc;
    }

    for (int level = 0; level < plan->max_levels; ++level) {
        FHTBoxList new_dirs;
        boxlist_init(&new_dirs);
        for (size_t idx = 0; idx < dir_boxes->size; ++idx) {
            FHTBox box = dir_boxes->data[idx];
            size_t d1 = box.i1 - box.i0 + 1;
            size_t d2 = box.j1 - box.j0 + 1;
            if (d1 * d2 > 4 * plan->min_dim_prod) {
                size_t ispl = box.i0;
                size_t jspl = box.j0;
                int s_rc = split_box(plan, &box, &ispl, &jspl);
                if (s_rc != CFASTFHT_OK) {
                    boxlist_free(&new_dirs);
                    return s_rc;
                }
                if (ispl > box.i0) {
                    s_rc = boxlist_push(loc_boxes, (FHTBox){box.i0, ispl - 1, box.j0, jspl - 1});
                    if (s_rc != CFASTFHT_OK) {
                        boxlist_free(&new_dirs);
                        return s_rc;
                    }
                    s_rc = boxlist_push(&new_dirs, (FHTBox){box.i0, ispl - 1, jspl, box.j1});
                    if (s_rc != CFASTFHT_OK) {
                        boxlist_free(&new_dirs);
                        return s_rc;
                    }
                }
                s_rc = boxlist_push(asy_boxes, (FHTBox){ispl, box.i1, jspl, box.j1});
                if (s_rc != CFASTFHT_OK) {
                    boxlist_free(&new_dirs);
                    return s_rc;
                }
                s_rc = boxlist_push(&new_dirs, (FHTBox){ispl, box.i1, box.j0, jspl - 1});
                if (s_rc != CFASTFHT_OK) {
                    boxlist_free(&new_dirs);
                    return s_rc;
                }
            } else {
                int push_rc = boxlist_push(&new_dirs, box);
                if (push_rc != CFASTFHT_OK) {
                    boxlist_free(&new_dirs);
                    return push_rc;
                }
            }
        }
        boxlist_free(dir_boxes);
        *dir_boxes = new_dirs;
    }

    FHTBoxList new_loc;
    FHTBoxList new_asy;
    boxlist_init(&new_loc);
    boxlist_init(&new_asy);

    size_t loc_threshold = plan->K_loc > 0 ? (size_t)plan->K_loc : 0;

    for (size_t idx = 0; idx < loc_boxes->size; ++idx) {
        FHTBox box = loc_boxes->data[idx];
        size_t d1 = box.i1 - box.i0 + 1;
        size_t d2 = box.j1 - box.j0 + 1;
        if (loc_threshold > 0 && MIN(d1, d2) < loc_threshold) {
            int rc2 = boxlist_push(dir_boxes, box);
            if (rc2 != CFASTFHT_OK) {
                boxlist_free(&new_loc);
                boxlist_free(&new_asy);
                return rc2;
            }
        } else {
            int rc2 = boxlist_push(&new_loc, box);
            if (rc2 != CFASTFHT_OK) {
                boxlist_free(&new_loc);
                boxlist_free(&new_asy);
                return rc2;
            }
        }
    }
    for (size_t idx = 0; idx < asy_boxes->size; ++idx) {
        FHTBox box = asy_boxes->data[idx];
        size_t d1 = box.i1 - box.i0 + 1;
        size_t d2 = box.j1 - box.j0 + 1;
        if (d1 * d2 < plan->min_dim_prod) {
            int rc2 = boxlist_push(dir_boxes, box);
            if (rc2 != CFASTFHT_OK) {
                boxlist_free(&new_loc);
                boxlist_free(&new_asy);
                return rc2;
            }
        } else {
            int rc2 = boxlist_push(&new_asy, box);
            if (rc2 != CFASTFHT_OK) {
                boxlist_free(&new_loc);
                boxlist_free(&new_asy);
                return rc2;
            }
        }
    }

    boxlist_free(loc_boxes);
    boxlist_free(asy_boxes);
    *loc_boxes = new_loc;
    *asy_boxes = new_asy;

    return CFASTFHT_OK;
}

static int allocate_plan_buffers(cfastfht_plan *plan) {
    plan->in_buffer = calloc(plan->m, sizeof(double complex));
    plan->out_buffer = calloc(plan->n, sizeof(double complex));
    plan->real_buffer_1 = calloc(plan->n, sizeof(double));
    plan->real_buffer_2 = calloc(plan->n, sizeof(double));
    size_t loc_terms = plan->K_loc >= 0 ? (size_t)plan->K_loc + 1 : 0;
    plan->cheb_buffer = loc_terms ? calloc(loc_terms, sizeof(double)) : NULL;
    plan->bessel_buffer_1 = loc_terms ? calloc(loc_terms, sizeof(double)) : NULL;
    plan->bessel_buffer_2 = NULL;

    if (!plan->in_buffer || !plan->out_buffer || !plan->real_buffer_1 ||
        !plan->real_buffer_2 || (loc_terms && (!plan->cheb_buffer || !plan->bessel_buffer_1))) {
        return CFASTFHT_ERR_ALLOC;
    }
    return CFASTFHT_OK;
}

static void free_plan_buffers(cfastfht_plan *plan) {
    free(plan->in_buffer);
    free(plan->out_buffer);
    free(plan->real_buffer_1);
    free(plan->real_buffer_2);
    free(plan->cheb_buffer);
    free(plan->bessel_buffer_1);
    free(plan->bessel_buffer_2);
}

static int setup_parameters(cfastfht_plan *plan, const cfastfht_options *options) {
    plan->min_dim_prod = (options && options->min_dim_prod)
        ? options->min_dim_prod
        : 10000;
    if (options && options->max_levels > 0) {
        plan->max_levels = options->max_levels;
    } else {
        double min_side = (double)MIN(plan->m, plan->n);
        double denom = (double)plan->min_dim_prod;
        if (denom < 1.0) denom = 1.0;
        double raw = log2((min_side * min_side) / denom);
        plan->max_levels = raw > 0 ? (int)floor(raw) : 0;
    }

    int user_Kasy = (options && options->K_asy >= -1) ? options->K_asy : INT_MIN;
    if (user_Kasy != INT_MIN) {
        plan->K_asy = user_Kasy;
    } else {
        plan->K_asy = MIN(10, (int)floor(fabs(plan->nu) / 5.0 + log10(1.0 / plan->tol) / 4.0 + 1.0));
    }
    if (plan->K_asy < 0) {
        plan->asy_coef_len = 0;
        plan->asy_coef = NULL;
    } else {
        plan->K_asy = MAX(0, plan->K_asy);
        plan->asy_coef_len = (size_t)(2 * plan->K_asy + 3);
        plan->asy_coef = calloc(plan->asy_coef_len, sizeof(double));
        if (!plan->asy_coef) {
            return CFASTFHT_ERR_ALLOC;
        }
        for (size_t k = 0; k < plan->asy_coef_len; ++k) {
            plan->asy_coef[k] = hankel_a((int)k, plan->nu);
        }
    }

    if (plan->K_asy >= 0) {
        double z_split = (options && !isnan(options->z_split))
            ? options->z_split
            : find_z_split(plan->nu, plan->K_asy, plan->tol);
        if (!isfinite(z_split)) {
            return CFASTFHT_ERR_BAD_INPUT;
        }
        plan->z_split = z_split;
    } else {
        plan->z_split = (options && !isnan(options->z_split))
            ? options->z_split
            : 0.0;
    }

    int user_Kloc = (options && options->K_loc >= -1) ? options->K_loc : INT_MIN;
    if (user_Kloc != INT_MIN) {
        plan->K_loc = user_Kloc;
    } else {
        plan->K_loc = compute_wimp_K(plan->nu, plan->K_asy >= 0 ? plan->K_asy : 0, plan->z_split, plan->tol);
    }
    if (plan->K_loc < 0) {
        plan->K_loc = -1;
    }

    return CFASTFHT_OK;
}

static void copy_vector(double *dest, const double *src, size_t len) {
    memcpy(dest, src, len * sizeof(double));
}

static int besselj_sequence(int lmax, double x, double *out) {
    if (lmax < 0) {
        return CFASTFHT_OK;
    }
    out[0] = j0(x);
    if (lmax == 0) {
        return CFASTFHT_OK;
    }
    out[1] = j1(x);
    if (fabs(x) < 1e-12) {
        for (int l = 2; l <= lmax; ++l) {
            out[l] = 0.0;
        }
        return CFASTFHT_OK;
    }
    for (int l = 1; l < lmax; ++l) {
        out[l + 1] = (2.0 * l / x) * out[l] - out[l - 1];
    }
    return CFASTFHT_OK;
}

static int add_loc_box(const cfastfht_plan *plan,
                       const double *cs,
                       double *gs,
                       const FHTBox *box) {
    if (plan->K_loc < 0) {
        return CFASTFHT_OK;
    }
    size_t w_start = box->i0 - 1;
    size_t w_count = box->i1 - box->i0 + 1;
    size_t r_start = box->j0 - 1;
    size_t r_count = box->j1 - box->j0 + 1;

    const double *rs = plan->rs + r_start;
    const double *ws = plan->ws + w_start;
    const double *cs_slice = cs + r_start;
    double *cheb = plan->cheb_buffer;
    double *bess = plan->bessel_buffer_1;
    double r_max = rs[r_count - 1];

    memset(cheb, 0, (plan->K_loc + 1) * sizeof(double));
    for (int l = 0; l <= plan->K_loc; ++l) {
        for (size_t idx = 0; idx < r_count; ++idx) {
            double ratio = clamp_value(rs[idx] / r_max, -1.0, 1.0);
            double theta = acos(ratio);
            double val = cos((double)l * theta);
            double term;
            if (l == 0) {
                term = (2.0 * val * val - 1.0) * cs_slice[idx];
            } else {
                term = 2.0 * (2.0 * val * val - 1.0) * cs_slice[idx];
            }
            cheb[l] += term;
        }
    }

    for (size_t wi = 0; wi < w_count; ++wi) {
        double arg = ws[wi] * r_max * 0.5;
        besselj_sequence(plan->K_loc, arg, bess);
        for (int l = 0; l <= plan->K_loc; ++l) {
            bess[l] *= bess[l];
            if (l % 2 == 1) {
                bess[l] *= -1.0;
            }
        }
        double accum = 0.0;
        for (int l = 0; l <= plan->K_loc; ++l) {
            accum += bess[l] * cheb[l];
        }
        gs[w_start + wi] += accum;
    }

    return CFASTFHT_OK;
}

static complex double cispi(double x) {
    double angle = M_PI * x;
    return cos(angle) + I * sin(angle);
}

static int add_asy_box(const cfastfht_plan *plan,
                       const double *cs,
                       double *gs,
                       const FHTBox *box) {
    if (plan->K_asy < 0 || plan->asy_coef == NULL) {
        return CFASTFHT_OK;
    }
    size_t w_start = box->i0 - 1;
    size_t w_count = box->i1 - box->i0 + 1;
    size_t r_start = box->j0 - 1;
    size_t r_count = box->j1 - box->j0 + 1;

    const double *rs = plan->rs + r_start;
    const double *ws = plan->ws + w_start;
    const double *cs_slice = cs + r_start;

    double complex *in_buf = plan->in_buffer;
    double complex *out_buf = plan->out_buffer;
    double *tmp1 = plan->real_buffer_1;
    double *tmp2 = plan->real_buffer_2;

    complex double phase = cispi(-plan->nu / 2.0 - 0.25);

    for (int l = 0; l <= plan->K_asy; ++l) {
        for (size_t k = 0; k < r_count; ++k) {
            double power = pow(rs[k], -2.0 * l - 1.0);
            double weight = cs_slice[k] * power * sqrt(rs[k]);
            in_buf[k] = weight + 0.0 * I;
        }
        int rc = finufft1d3((long long)r_count,
                             rs,
                             in_buf,
                             +1,
                             plan->tol,
                             (long long)w_count,
                             ws,
                             out_buf,
                             NULL);
        if (rc != 0) {
            return CFASTFHT_ERR_FINUFFT;
        }
        for (size_t j = 0; j < w_count; ++j) {
            out_buf[j] *= phase;
            tmp1[j] = creal(out_buf[j]);
            tmp2[j] = pow(ws[j], -2.0 * l - 1.0) * sqrt(ws[j]);
        }
        double coeff = sqrt(2.0 / M_PI) * (l % 2 == 0 ? 1.0 : -1.0) * plan->asy_coef[2 * l + 1];
        for (size_t j = 0; j < w_count; ++j) {
            gs[w_start + j] += coeff * tmp1[j] * tmp2[j];
        }

        for (size_t k = 0; k < r_count; ++k) {
            double power = pow(rs[k], -2.0 * l - 2.0);
            double weight = cs_slice[k] * power * sqrt(rs[k]);
            in_buf[k] = weight + 0.0 * I;
        }
        rc = finufft1d3((long long)r_count,
                        rs,
                        in_buf,
                        +1,
                        plan->tol,
                        (long long)w_count,
                        ws,
                        out_buf,
                        NULL);
        if (rc != 0) {
            return CFASTFHT_ERR_FINUFFT;
        }
        for (size_t j = 0; j < w_count; ++j) {
            out_buf[j] *= phase;
            tmp1[j] = cimag(out_buf[j]);
            tmp2[j] = pow(ws[j], -2.0 * l - 2.0) * sqrt(ws[j]);
        }
        coeff = sqrt(2.0 / M_PI) * (l % 2 == 0 ? 1.0 : -1.0) * plan->asy_coef[2 * l + 2];
        for (size_t j = 0; j < w_count; ++j) {
            gs[w_start + j] -= coeff * tmp1[j] * tmp2[j];
        }
    }

    return CFASTFHT_OK;
}

static int add_dir_box(const cfastfht_plan *plan,
                       const double *cs,
                       double *gs,
                       const FHTBox *box) {
    size_t w_start = box->i0 - 1;
    size_t w_end = box->i1;
    size_t r_start = box->j0 - 1;
    size_t r_end = box->j1;

    const double *rs = plan->rs;
    const double *ws = plan->ws;

    for (size_t j = w_start; j < w_end; ++j) {
        double wval = ws[j];
        double accum = 0.0;
        for (size_t k = r_start; k < r_end; ++k) {
            accum += cs[k] * j0(wval * rs[k]);
        }
        gs[j] += accum;
    }
    return CFASTFHT_OK;
}

static int apply_plan(const cfastfht_plan *plan,
                      const double *cs,
                      double *gs) {
    int rc;
    for (size_t idx = 0; idx < plan->loc_boxes.size; ++idx) {
        rc = add_loc_box(plan, cs, gs, &plan->loc_boxes.data[idx]);
        if (rc != CFASTFHT_OK) return rc;
    }
    for (size_t idx = 0; idx < plan->asy_boxes.size; ++idx) {
        rc = add_asy_box(plan, cs, gs, &plan->asy_boxes.data[idx]);
        if (rc != CFASTFHT_OK) return rc;
    }
    for (size_t idx = 0; idx < plan->dir_boxes.size; ++idx) {
        rc = add_dir_box(plan, cs, gs, &plan->dir_boxes.data[idx]);
        if (rc != CFASTFHT_OK) return rc;
    }
    return CFASTFHT_OK;
}

cfastfht_plan *cfastfht_plan_create(double nu,
                                    const double *rs,
                                    size_t rs_len,
                                    const double *ws,
                                    size_t ws_len,
                                    double tol,
                                    const cfastfht_options *options) {
    if (rs_len == 0 || ws_len == 0 || !rs || !ws) {
        set_last_error("rs/ws arrays must be non-empty");
        return NULL;
    }
    if (fabs(nu) > 1e-12) {
        set_last_error("current C backend only supports nu = 0");
        return NULL;
    }
    if (tol <= 0.0 || tol >= 1e-2) {
        set_last_error("tolerance must be in (0, 1e-2)");
        return NULL;
    }
    for (size_t i = 0; i < rs_len; ++i) {
        if (!(rs[i] > 0.0)) {
            set_last_error("rs must be positive");
            return NULL;
        }
    }
    for (size_t i = 0; i < ws_len; ++i) {
        if (!(ws[i] > 0.0)) {
            set_last_error("ws must be positive");
            return NULL;
        }
    }
    if (!is_sorted_strict(rs, rs_len) || !is_sorted_strict(ws, ws_len)) {
        set_last_error("rs and ws must be strictly increasing");
        return NULL;
    }

    cfastfht_plan *plan = calloc(1, sizeof(cfastfht_plan));
    if (!plan) {
        set_last_error("failed to allocate plan");
        return NULL;
    }

    plan->nu = 0.0;
    plan->tol = tol;
    plan->m = rs_len;
    plan->n = ws_len;
    plan->rs = calloc(rs_len, sizeof(double));
    plan->ws = calloc(ws_len, sizeof(double));
    if (!plan->rs || !plan->ws) {
        set_last_error("failed to allocate grid storage");
        cfastfht_plan_destroy(plan);
        return NULL;
    }
    copy_vector(plan->rs, rs, rs_len);
    copy_vector(plan->ws, ws, ws_len);

    boxlist_init(&plan->loc_boxes);
    boxlist_init(&plan->asy_boxes);
    boxlist_init(&plan->dir_boxes);

    int rc = setup_parameters(plan, options);
    if (rc != CFASTFHT_OK) {
        set_last_error("failed to setup parameters");
        cfastfht_plan_destroy(plan);
        return NULL;
    }

    rc = allocate_plan_buffers(plan);
    if (rc != CFASTFHT_OK) {
        set_last_error("failed to allocate scratch buffers");
        cfastfht_plan_destroy(plan);
        return NULL;
    }

    rc = generate_boxes(plan, &plan->loc_boxes, &plan->asy_boxes, &plan->dir_boxes);
    if (rc != CFASTFHT_OK) {
        set_last_error("failed to generate boxes");
        cfastfht_plan_destroy(plan);
        return NULL;
    }

    return plan;
}

void cfastfht_plan_destroy(cfastfht_plan *plan) {
    if (!plan) {
        return;
    }
    free(plan->rs);
    free(plan->ws);
    free(plan->asy_coef);
    boxlist_free(&plan->loc_boxes);
    boxlist_free(&plan->asy_boxes);
    boxlist_free(&plan->dir_boxes);
    free_plan_buffers(plan);
    free(plan);
}

int cfastfht_plan_execute(const cfastfht_plan *plan,
                          const double *cs,
                          double *out) {
    if (!plan || !cs || !out) {
        set_last_error("null pointer passed to execute");
        return CFASTFHT_ERR_BAD_INPUT;
    }
    memset(out, 0, plan->n * sizeof(double));
    int rc = apply_plan(plan, cs, out);
    if (rc != CFASTFHT_OK) {
        set_last_error("execution failed (%s)", cfastfht_strerror(rc));
    }
    return rc;
}

int cfastfht_plan_execute_batch(const cfastfht_plan *plan,
                                const double *coeffs,
                                size_t coeff_stride,
                                double *out,
                                size_t out_stride,
                                size_t batch_size) {
    if (!plan || !coeffs || !out) {
        set_last_error("null pointer passed to execute_batch");
        return CFASTFHT_ERR_BAD_INPUT;
    }
    if (coeff_stride < plan->m || out_stride < plan->n) {
        set_last_error("strides are too small");
        return CFASTFHT_ERR_BAD_INPUT;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        const double *cs = coeffs + b * coeff_stride;
        double *dest = out + b * out_stride;
        memset(dest, 0, plan->n * sizeof(double));
        int rc = apply_plan(plan, cs, dest);
        if (rc != CFASTFHT_OK) {
            set_last_error("batch execution failed on index %zu", b);
            return rc;
        }
    }
    return CFASTFHT_OK;
}
