#include <iostream>
#include <cmath>
#include <string>
#include <omp.h>
#include <vector>
#include <mkl.h>
#include <mkl_vsl.h>

double f(double* v, int size) {
    double ans = 1;
    for (int i = 0; i < size; i++) {
        ans *= sin(v[i]);
    }
    return ans;
}

double left_rectangle(double(*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = 0;
    for (double point = l; point < r; point += step) {
        ans += func(point);
    }
    return ans * step;
}

double left_rectangle_parallel(double (*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = 0;
    double point;
#pragma omp parallel for reduction(+ : ans)
    for (int i = 0; i < count_point; i++) {
        point = l + i * step;
        ans += func(point);
    }
    return ans * step;
}

double right_rectangle(double(*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = 0;
    for (double point = l + step; point <= r; point += step) {
        ans += func(point);
    }
    return ans * step;
}

double right_rectangle_parallel(double (*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = 0;
    double point;
#pragma omp parallel for reduction(+ : ans)
    for (int i = 1; i < count_point + 1; i++) {
        point = l + i * step;
        ans += func(point);
    }
    return ans * step;
}

double sentr_rectangle(double(*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = 0;
    for (double point = l + step / 2; point < r; point += step) {
        ans += func(point);
    }
    return ans * step;
}

double sentr_rectangle_parallel(double (*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = 0;
    double point;
#pragma omp parallel for reduction(+ : ans)
    for (int i = 0; i < count_point; i++) {
        point = l + (i + 0.5) * step;
        ans += func(point);
    }
    return ans * step;
}

double trapezoid(double(*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = (func(l) + func(r)) / 2;
    double point = step;
    for (int i = 1; i < count_point; i++) {
        ans += func(point);
        point += step;
    }
    return ans * step;
}

double trapezoid_parallel(double(*func)(double x), double l, double r, int count_point) {
    const double step = (r - l) / count_point;
    double ans = (func(l) + func(r)) / 2;
    double point;
#pragma omp parallel for reduction(+ : ans)
    for (int i = 1; i <= count_point - 1; i++) {
        point = l + (i)*step;
        ans += func(point);
    }
    return ans * step;
}

double base_montekarlo(double(*func)(double x), double l, double r, int count_point) {
    double ans = 0;
    double* p = new double[1];
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT2203, 777);
    for (int i = 0; i < count_point; i++) {
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, p, l, r);
        ans += func(*p);
    }
    vslDeleteStream(&stream);
    return ans * (r - l) / count_point;
}

double base_montekarlo_parallel(double(*func)(double x), double l, double r, int count_point) {
    double ans = 0;
    double p = 0;
    int th = omp_get_max_threads();
    int count_point_old = count_point;
    count_point /= th;
#pragma omp parallel reduction(+ : ans)
    {
        int th_n = omp_get_thread_num();
        VSLStreamStatePtr stream_n;
        vslNewStream(&stream_n, VSL_BRNG_MT19937, 777 + th_n);
        for (int i = 0; i < count_point; i++) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_n, 1, &p, l, r);
            ans += func(p);
        }
        vslDeleteStream(&stream_n);
    }
    return ans * (r - l) / count_point_old;
}

//Двумерное
double sentr_rectangle_2(double (*func)(double, double), double l_x, double r_x, double l_y, double r_y, int c_x, int c_y) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
    for (int i = 0; i < c_x; i++) {
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            ans += func(p_x, p_y);
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y;
}

double sentr_rectangle_2_parallel(double (*func)(double, double), double l_x, double r_x, double l_y, double r_y, int c_x, int c_y) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
#pragma omp parallel for reduction(+ : ans) private(p_x)
    for (int i = 0; i < c_x; i++) {
        p_x = l_x + step_x * (i + 0.5);
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            ans += func(p_x, p_y);
            p_y += step_y;
        }
    }
    return ans * step_x * step_y;
}

double trapezoid_2(double (*func)(double, double), double l_x, double r_x, double l_y, double r_y, int c_x, int c_y) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double k;
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        for (int j = 0; j <= c_y; j++) {
            k = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))));
            k = 1.0 / (1 << (2 - (int) k));
            ans += k * func(p_x, p_y);
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y;
}

double trapezoid_2_parallel(double (*func)(double, double), double l_x, double r_x, double l_y, double r_y, int c_x, int c_y) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double k;
#pragma omp parallel for reduction(+ : ans) private(p_x)
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        p_x = l_x + step_x * i;
        for (int j = 0; j <= c_y; j++) {
            k = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))));
            k = 1.0 / (1 << (2 - (int)k));
            ans += k * func(p_x, p_y);
            p_y += step_y;
        }
    }
    return ans * step_x * step_y;
}
//Трехмерное
double sentr_rectangle_3(double (*func)(double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, int c_x, int c_y, int c_z) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
    double p_z = l_z + step_z / 2;
    for (int i = 0; i < c_x; i++) {
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            p_z = l_z + step_z / 2;
            for (int k = 0; k < c_z; k++) {
                ans += func(p_x, p_y, p_z);
                p_z += step_z;
            }
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y * step_z;
}
double sentr_rectangle_3_parallel(double (*func)(double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, int c_x, int c_y, int c_z) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
    double p_z = l_z + step_z / 2;
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y)
    for (int i = 0; i < c_x; i++) {
        p_x = l_x + (i + 0.5) * step_x;
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            p_z = l_z + step_z / 2;
            for (int k = 0; k < c_z; k++) {
                ans += func(p_x, p_y, p_z);
                p_z += step_z;
            }
            p_y += step_y;
        }
    }
    return ans * step_x * step_y * step_z;
}

double trapezoid_3(double (*func)(double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, int c_x, int c_y, int c_z) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double p_z = l_z;
    double kf;
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        for (int j = 0; j <= c_y; j++) {
            p_z = l_z;
            for (int k = 0; k <= c_z; k++) {
                kf = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))) + ((k != 0) && (k != (c_z))));
                kf = 1.0 / (1 << (3 - (int)kf));
                ans += kf * func(p_x, p_y, p_z);
                p_z += step_z;
            }
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y * step_z;
}

double trapezoid_3_parallel(double (*func)(double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, int c_x, int c_y, int c_z) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double p_z = l_z;
    double kf;
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y)
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        p_x = l_x + step_x * i;
        for (int j = 0; j <= c_y; j++) {
            p_z = l_z;
            for (int k = 0; k <= c_z; k++) {
                kf = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))) + ((k != 0) && (k != (c_z))));
                kf = 1.0 / (1 << (3 - (int)kf));
                ans += kf * func(p_x, p_y, p_z);
                p_z += step_z;
            }
            p_y += step_y;
        }
    }
    return ans * step_x * step_y * step_z;
}
// 4-мерное
double sentr_rectangle_4(double (*func)(double, double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
    double p_z = l_z + step_z / 2;
    double p_w = l_w + step_w / 2;
    for (int i = 0; i < c_x; i++) {
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            p_z = l_z + step_z / 2;
            for (int k = 0; k < c_z; k++) {
                p_w = l_w + step_w / 2;
                for (int e = 0; e < c_w; e++) {
                    ans += func(p_x, p_y, p_z, p_w);
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y * step_z * step_w;
}
double sentr_rectangle_4_parallel(double (*func)(double, double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
    double p_z = l_z + step_z / 2;
    double p_w = l_w + step_w / 2;
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y, p_z)
    for (int i = 0; i < c_x; i++) {
        p_x = l_x + (i + 0.5) * step_x;
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            p_z = l_z + step_z / 2;
            for (int k = 0; k < c_z; k++) {
                p_w = l_w + step_w / 2;
                for (int e = 0; e < c_w; e++) {
                    ans += func(p_x, p_y, p_z, p_w);
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
    }
    return ans * step_x * step_y * step_z * step_w;
}

double trapezoid_4(double (*func)(double, double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double p_z = l_z;
    double p_w = l_w;
    double kf;
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        for (int j = 0; j <= c_y; j++) {
            p_z = l_z;
            for (int k = 0; k <= c_z; k++) {
                p_w = l_w;
                for (int e = 0; e <= c_w; e++) {
                    kf = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))) + ((k != 0) && (k != (c_z))) + ((e != 0) && (e != (c_w))));
                    kf = 1.0 / (1 << (4 - (int)kf));
                    ans += kf * func(p_x, p_y, p_z, p_w);
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y * step_z * step_w;
}
double trapezoid_4_parallel(double (*func)(double, double, double, double), double l_x, double r_x, double l_y, double r_y,
    double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double p_z = l_z;
    double p_w = l_w;
    double kf;
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y, p_z)
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        p_x = l_x + i * step_x;
        for (int j = 0; j <= c_y; j++) {
            p_z = l_z;
            for (int k = 0; k <= c_z; k++) {
                p_w = l_w;
                for (int e = 0; e <= c_w; e++) {
                    kf = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))) + ((k != 0) && (k != (c_z))) + ((e != 0) && (e != (c_w))));
                    kf = 1.0 / (1 << (4 - (int)kf));
                    ans += kf * func(p_x, p_y, p_z, p_w);
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
    }
    return ans * step_x * step_y * step_z * step_w;
}
// 5-мерное
double sentr_rectangle_5(double (*func)(double, double, double, double, double), double l_x, double r_x,
    double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
    double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double step_q = (r_q - l_q) / c_q;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
    double p_z = l_z + step_z / 2;
    double p_w = l_w + step_w / 2;
    double p_q = l_q + step_q / 2;
    for (int i = 0; i < c_x; i++) {
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            p_z = l_z + step_z / 2;
            for (int k = 0; k < c_z; k++) {
                p_w = l_w + step_w / 2;
                for (int e = 0; e < c_w; e++) {
                    p_q = l_q + step_q / 2;
                    for (int ee = 0; ee < c_q; ee++) {
                        ans += func(p_x, p_y, p_z, p_w, p_q);
                        p_q += step_q;
                    }
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y * step_z * step_w * step_q;
}
double sentr_rectangle_5_parallel(double (*func)(double, double, double, double, double), double l_x, double r_x,
    double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
    double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double step_q = (r_q - l_q) / c_q;
    double ans = 0;
    double p_x = l_x + step_x / 2;
    double p_y = l_y + step_y / 2;
    double p_z = l_z + step_z / 2;
    double p_w = l_w + step_w / 2;
    double p_q = l_q + step_q / 2;
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y, p_z, p_w)
    for (int i = 0; i < c_x; i++) {
        p_x = l_x + step_x * (i + 0.5);
        p_y = l_y + step_y / 2;
        for (int j = 0; j < c_y; j++) {
            p_z = l_z + step_z / 2;
            for (int k = 0; k < c_z; k++) {
                p_w = l_w + step_w / 2;
                for (int e = 0; e < c_w; e++) {
                    p_q = l_q + step_q / 2;
                    for (int ee = 0; ee < c_q; ee++) {
                        ans += func(p_x, p_y, p_z, p_w, p_q);
                        p_q += step_q;
                    }
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
    }
    return ans * step_x * step_y * step_z * step_w * step_q;
}

double trapezoid_5(double (*func)(double, double, double, double, double), double l_x, double r_x,
    double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
    double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double step_q = (r_q - l_q) / c_q;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double p_z = l_z;
    double p_w = l_w;
    double p_q = l_q;
    double kf;
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        for (int j = 0; j <= c_y; j++) {
            p_z = l_z;
            for (int k = 0; k <= c_z; k++) {
                p_w = l_w;
                for (int e = 0; e <= c_w; e++) {
                    p_q = l_q;
                    for (int ee = 0; ee <= c_q; ee++) {
                        kf = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))) + ((k != 0) && (k != (c_z))) + ((e != 0) && (e != (c_w))) + ((ee != 0) && (j != (c_q))));
                        kf = 1.0 / (1 << (5 - (int)kf));
                        ans += kf * func(p_x, p_y, p_z, p_w, p_q);
                        p_q += step_q;
                    }
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
        p_x += step_x;
    }
    return ans * step_x * step_y * step_z * step_w * step_q;
}

double trapezoid_5_parallel(double (*func)(double, double, double, double, double), double l_x, double r_x,
    double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
    double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q) {
    double step_x = (r_x - l_x) / c_x;
    double step_y = (r_y - l_y) / c_y;
    double step_z = (r_z - l_z) / c_z;
    double step_w = (r_w - l_w) / c_w;
    double step_q = (r_q - l_q) / c_q;
    double ans = 0;
    double p_x = l_x;
    double p_y = l_y;
    double p_z = l_z;
    double p_w = l_w;
    double p_q = l_q;
    double kf;
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y, p_z, p_w)
    for (int i = 0; i <= c_x; i++) {
        p_y = l_y;
        p_x = l_x + i * step_x;
        for (int j = 0; j <= c_y; j++) {
            p_z = l_z;
            for (int k = 0; k <= c_z; k++) {
                p_w = l_w;
                for (int e = 0; e <= c_w; e++) {
                    p_q = l_q;
                    for (int ee = 0; ee <= c_q; ee++) {
                        kf = (((i != 0) && (i != (c_x))) + ((j != 0) && (j != (c_y))) + ((k != 0) && (k != (c_z))) + ((e != 0) && (e != (c_w))) + ((ee != 0) && (j != (c_q))));
                        kf = 1.0 / (1 << (5 - (int)kf));
                        ans += kf * func(p_x, p_y, p_z, p_w, p_q);
                        p_q += step_q;
                    }
                    p_w += step_w;
                }
                p_z += step_z;
            }
            p_y += step_y;
        }
    }
    return ans * step_x * step_y * step_z * step_w * step_q;
}
// n-мерное
double base_montekarlo_n(int n, double (*func)(double *, int), std::vector<double> l, std::vector<double> r, int c) {
    double ans = 0;
    double* p = new double[n];
    double k = 1;
    for (int i = 0; i < n; i++) {
        k *= (r[i] - l[i]);
    }
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT2203, 777);
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < n; j++) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, p+j, l[j], r[j]);
        }
        ans += func(p, n);
    }
    vslDeleteStream(&stream);
    return k * ans / c;
}

double base_montekarlo_n_parallel(int n, double (*func)(double*, int), std::vector<double> l, std::vector<double> r, int c) {
    double ans = 0;
    int th = omp_get_max_threads();
    int c_old = c;
    c /= th;
    double k = 1;
    for (int i = 0; i < n; i++) {
        k *= (r[i] - l[i]);
    }
    #pragma omp parallel reduction(+ : ans)
    {
        int th_n = omp_get_thread_num();
        VSLStreamStatePtr stream_n;
        vslNewStream(&stream_n, VSL_BRNG_MT19937, 777 + th_n);
        double* p = new double[n];
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < n; j++) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_n, 1, &p[j], l[j], r[j]);
            }
            ans += func(p, n);
        }
        vslDeleteStream(&stream_n);
    }
    return ans * k / c_old;
}

void start_res(long long int cp_max, int step, int parallel, int n) {
    double ans = 0;
    if (n == 1) {
        double(*func)(double x) = sin;
        double right_ans = 1 - cos(1);
        for (int c_p = 1; c_p < cp_max; c_p += step) {
            ans = right_rectangle(func, 0, 1, c_p);
            std::cout << abs(ans - right_ans) << " ";
        }
    }
    if (n == 2) {
        auto func{ [](double x, double y) {return sin(x) * sin(y); } };
        double right_ans = (1 - cos(1)) * (1 - cos(1));
        for (int c_p = 1; c_p < cp_max; c_p += step) {
            ans = sentr_rectangle_2(func, 0, 1, 0, 1, c_p, c_p);
            std::cout << abs(ans - right_ans) << " ";
        }
    }
    if (n == 3) {
        auto func{ [](double x, double y, double z) {return sin(x) * sin(y) * sin(z); } };
        double right_ans = (1 - cos(1)) * (1 - cos(1)) * (1 - cos(1));
        for (int c_p = 1; c_p < cp_max; c_p += step) {
            ans = sentr_rectangle_3(func, 0, 1, 0, 1, 0, 1, c_p, c_p, c_p);
            std::cout << abs(ans - right_ans) << " ";
        }
    }
    if (n == 4) {
        auto func{ [](double x, double y, double z, double w) {return sin(x) * sin(y) * sin(z) * sin(w); } };
        double right_ans = (1 - cos(1)) * (1 - cos(1)) * (1 - cos(1)) * (1 - cos(1));
        for (int c_p = 1; c_p < cp_max; c_p += step) {
            ans = sentr_rectangle_4(func, 0, 1, 0, 1, 0, 1, 0, 1, c_p, c_p, c_p, c_p);
            std::cout << abs(ans - right_ans) << " ";
        }
    }
    if (n == 5) {
        auto func{ [](double x, double y, double z, double w, double q) {return sin(x) * sin(y) * sin(z) * sin(w) * sin(q); } };
        double right_ans = (1 - cos(1)) * (1 - cos(1)) * (1 - cos(1)) * (1 - cos(1)) * (1 - cos(1));
        for (int c_p = 1; c_p < cp_max; c_p += step) {
            ans = sentr_rectangle_5(func, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, c_p, c_p, c_p, c_p, c_p);
            std::cout << abs(ans - right_ans) << " ";
        }
    }
    if (n > 5) {
        auto func = *f;
        double right_ans = 1;
        auto l = std::vector<double>(n);
        auto r = std::vector<double>(n);
        auto c = std::vector<int>(n);
        for (int i = 0; i < n; i++) {
            l[i] = 0;
            r[i] = 1;
            right_ans *= (1 - cos(1));
        }
        for (int c_p = 1; c_p < cp_max; c_p += step) {
            ans = base_montekarlo_n(n, func, l, r, c_p);
            std::cout << abs(ans - right_ans) << " ";
        }
    }
}
