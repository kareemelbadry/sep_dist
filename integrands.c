/* Helper function for quick integration. Compile using 
 * gcc -shared -o integrands.so -fPIC integrands.c 
 * add -std=c99 if you get a compilation problem */

#include <math.h>
#include <stdlib.h>

double trapz(int n, double *y, double *x){
    double intgrl = 0;
    for(int i = 0; i+1 < n; i++){
        intgrl += (x[i+1] - x[i])*(y[i+1] + y[i])/2;
    }
    return intgrl;    
}

double* linspace(int n, double x_min, double x_max){
    double* _x = malloc(n*sizeof(double));
    double current = x_min;
    for(int i = 0; i < n; i++){
        _x[i] = current;
        current += (x_max - x_min)/(n - 1); 
    }
    return _x;
}

double* logspace(int n, double logx_min, double logx_max){
    double* _logxx = linspace(n, logx_min, logx_max);
    double* _x = malloc(n*sizeof(double));
    for(int i = 0; i < n; i++){
        _x[i] = pow(10, _logxx[i]);
    }
    free(_logxx);
    return _x;
}

/* the integrand of what I called I_1 in the math document*/
double integrand_einasto_x(double x, double x0, double alpha,
    double beta){
    return exp(-pow((x/x0), alpha))/(1 + pow(x, -beta));
}


/* what I called I_1 in the math document. n is the number of steps
to take in integrating, and logxx is the array of steps. I.e., xx goes from 
x_min to x_max (in log steps). */
double numerically_integrate_einasto_x(double x0, double alpha,
    double beta, double *xx, int n){
    double integrand[n];
    for(int i = 0; i < n; i++){
        integrand[i] = integrand_einasto_x(xx[i], x0, alpha, beta);
    }
    double intgrl = trapz(n, integrand, xx);
    return intgrl;
}

/* what I called "I" in the math document. */
double numerically_integrate_einasto_s(double d_pc, double theta_0,
    double s0, double alpha, double beta, double logs_min, double logs_max){
    double x0 = s0/(d_pc*theta_0);
    int m = 100; // number of steps for integral. 
    double *xx = logspace(m, logs_min-log10(d_pc*theta_0), logs_max-log10(d_pc*theta_0));
    double I1 = numerically_integrate_einasto_x(x0, alpha, beta, xx, m);
    free(xx);
    return I1*d_pc*theta_0;
}

/* read in a whole array of parameters (one for each binary) and return an array of weights.  
These "weights" are Eq 9 of the paper. n is the length of the array of binaries. 
w_i is the array of weights that will be filled by this function. */
void get_weights_einasto(int n, double s0, double alpha, double *w_i, double *betas, 
    double* theta0s, double *d_pc, double logs_min, double logs_max){
    for(int i = 0; i<n; i++){
        double wi = numerically_integrate_einasto_s(d_pc[i], theta0s[i],
            s0, alpha, betas[i], logs_min, logs_max);
        w_i[i] = wi;
    }
}
