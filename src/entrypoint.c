// Required so R can find the Rust-registered routines.
#include <R_ext/Rdynload.h>

void R_init_Remeans_extendr(DllInfo *dll);

void R_init_Remeans(DllInfo *dll) {
    R_init_Remeans_extendr(dll);
}
