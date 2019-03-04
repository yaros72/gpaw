#include "extensions.h"
//#include <stdlib.h>

PyObject *pc_potential(PyObject *self, PyObject *args)
{
    PyArrayObject* beg_v_obj;
    PyArrayObject* h_v_obj;
    PyArrayObject* q_p_obj;
    PyArrayObject* R_pv_obj;
    double rc;
    double rc2;
    double width;
    PyArrayObject* vext_G_obj;
    PyArrayObject* dcom_pv_obj;
    PyArrayObject* rhot_G_obj = 0;
    PyArrayObject* F_pv_obj = 0;
    if (!PyArg_ParseTuple(args, "OOOOdddOO|OO", &beg_v_obj, &h_v_obj, &q_p_obj,
                          &R_pv_obj, &rc, &rc2, &width,
                          &vext_G_obj, &dcom_pv_obj, &rhot_G_obj, &F_pv_obj))
    return NULL;

    const long *beg_v = PyArray_DATA(beg_v_obj);
    const double *h_v = PyArray_DATA(h_v_obj);
    const double *q_p = PyArray_DATA(q_p_obj);
    const double *R_pv = PyArray_DATA(R_pv_obj);
    const double *dcom_pv = 0;
    if ((PyObject*)dcom_pv_obj != Py_None)
        dcom_pv = PyArray_DATA(dcom_pv_obj);
    double *vext_G = PyArray_DATA(vext_G_obj);

    int np = PyArray_DIM(R_pv_obj, 0);
    npy_intp* n = PyArray_DIMS(vext_G_obj);

    const double* rhot_G = 0;
    double* F_pv = 0;
    double dV = 0.0;
    if (F_pv_obj != 0) {
        // Handle the two extra arguments for the force calculation:
        rhot_G = PyArray_DATA(rhot_G_obj);
        F_pv = PyArray_DATA(F_pv_obj);
        dV = h_v[0] * h_v[1] * h_v[2];
    }

    double rc12 = rc2 - width;

    for (int i = 0; i < n[0]; i++) {
        double x = (beg_v[0] + i) * h_v[0];
        for (int j = 0; j < n[1]; j++) {
            double y = (beg_v[1] + j) * h_v[1];
            int ij = (i * n[1] + j) * n[2];
            for (int k = 0; k < n[2]; k++) {
                double z = (beg_v[2] + k) * h_v[2];
                for (int p = 0; p < np; p++) {
                    const double* R_v = R_pv + 3 * p;
                    double dx = R_v[0] - x;
                    double dy = R_v[1] - y;
                    double dz = R_v[2] - z;
                    double d  = sqrt(dx * dx + dy * dy + dz * dz);
                    double dc, dxc, dyc, dzc;
                    if (dcom_pv == 0) {
                        dc = d;
                        dxc = dx;
                        dyc = dy;
                        dzc = dz;
                    } else {
                        const double* dcom_v = dcom_pv + 3 * p;
                        dxc = dcom_v[0];
                        dyc = dcom_v[1];
                        dzc = dcom_v[2];
                        dc = sqrt(dxc * dxc + dyc * dyc + dzc * dzc);
                    }
                    int G = ij + k;
                    if (F_pv == 0) {
                        // Calculate potential:
                        double v;
                        if (rc < 0.0)
                           v = (q_p[p] * (d * d * d * d - rc * rc * rc * rc) /
                                (d * d * d * d * d + rc * rc * rc * rc * rc));
                        else
                            if (dc > rc2)
                                v = 0.0;
                            else if (dc > rc12) {
                                double x = (dc - rc12) / width;
                                v = q_p[p] * (1 - x * x * (3 - 2 * x)) / d;
                            }
                            else if (d > rc)
                                v = q_p[p] / d;
                            else {
                                double s = d / rc;
                                double s2 = s * s;
                                v = q_p[p] * (3.28125 +
                                              s2 * (-5.46875 +
                                                    s2 * (4.59375 +
                                                          s2 * -1.40625))) / rc;
                            }
                        vext_G[G] -= v;
                    }
                    else {
                        // Calculate forces:
                        double w;  // -(dv/dr)/r
                        double o = 0.0;
                        if (rc < 0.0) {
                            double x = (d * d * d * d * d +
                                        rc * rc * rc * rc * rc);
                            w = ((d * d * d * d - rc * rc * rc * rc) /
                                 (x * x) * 5 * d * d * d -
                                 4 * d * d / x);
                        }
                        else
                            if (dc > rc2)
                                w = 0.0;
                            else if (dc > rc12) {
                                double x = (dc - rc12) / width;
                                w = (1 - x * x * (3 - 2 * x)) / (d * d * d);
                                o = 6 * x * (1 - x) / (width * dc * d);
                            }
                            else if (d > rc)
                                w = 1 / (d * d * d);
                            else {
                                double s = d / rc;
                                double s2 = s * s;
                                w = (-2 * (-5.46875 +
                                           s2 * (2 * 4.59375 +
                                                 s2 * 3 * -1.40625)) /
                                     (rc * rc * rc));
                            }
                        w *= q_p[p] * rhot_G[G] * dV;
                        o *= q_p[p] * rhot_G[G] * dV;
                        double* F_v = F_pv + 3 * p;
                        F_v[0] -= w * dx + o * dxc;
                        F_v[1] -= w * dy + o * dyc;
                        F_v[2] -= w * dz + o * dzc;
                    }
                }
            }
        }
    }
    Py_RETURN_NONE;
}
