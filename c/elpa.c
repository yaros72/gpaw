#if defined(GPAW_WITH_SL) && defined(PARALLEL) && defined(GPAW_WITH_ELPA)

#include "extensions.h"
#include <elpa/elpa.h>
#include <mpi.h>
#include "mympi.h"

elpa_t* unpack_handleptr(PyObject* handle_obj)
{
    elpa_t* elpa = (elpa_t *)PyArray_DATA((PyArrayObject *)handle_obj);
    return elpa;
}

elpa_t unpack_handle(PyObject* handle_obj)
{
    elpa_t* elpa = unpack_handleptr(handle_obj);
    return *elpa;
}

PyObject* checkerr(int err)
{
    if(err != ELPA_OK) {
        const char * errmsg = elpa_strerr(err);
        PyErr_SetString(PyExc_RuntimeError, errmsg);
        return NULL;
    }
    Py_RETURN_NONE;
}

PyObject* pyelpa_set(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    char* varname;
    int value;
    if (!PyArg_ParseTuple(args, "Osi",
                          &handle_obj,
                          &varname,
                          &value)) {
        return NULL;
    }
    elpa_t handle = unpack_handle(handle_obj);
    int err;
    elpa_set(handle, varname, value, &err);
    return checkerr(err);
}

PyObject* pyelpa_allocate(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t *handle = unpack_handleptr(handle_obj);
    int err = 0;
    handle[0] = elpa_allocate(&err);
    return checkerr(err);
}

PyObject* pyelpa_setup(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);
    int err = elpa_setup(handle);
    return checkerr(err);
}


PyObject* pyelpa_set_comm(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    PyObject *gpaw_comm_obj;

    if(!PyArg_ParseTuple(args, "OO", &handle_obj,
                         &gpaw_comm_obj))
        return NULL;
    elpa_t handle = unpack_handle(handle_obj);
    MPIObject *gpaw_comm = (MPIObject *)gpaw_comm_obj;
    MPI_Comm comm = gpaw_comm->comm;
    int fcomm = MPI_Comm_c2f(comm);
    int err;
    elpa_set(handle, "mpi_comm_parent", fcomm, &err);
    return checkerr(err);
}

PyObject* pyelpa_constants(PyObject *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, ""))
        return NULL;
    return Py_BuildValue("iii",
                         ELPA_OK,
                         ELPA_SOLVER_1STAGE,
                         ELPA_SOLVER_2STAGE);
}


PyObject* pyelpa_diagonalize(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    PyArrayObject *A_obj, *C_obj, *eps_obj;

    if (!PyArg_ParseTuple(args,
                          "OOOO",
                          &handle_obj,
                          &A_obj,
                          &C_obj,
                          &eps_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);

    double *a = (double*)PyArray_DATA(A_obj);
    double *ev = (double*)PyArray_DATA(eps_obj);
    double *q = (double*)PyArray_DATA(C_obj);

    int err;
    elpa_eigenvectors(handle, a, ev, q, &err);
    return checkerr(err);
}

PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    PyArrayObject *A_obj, *S_obj, *C_obj, *eps_obj;
    int is_already_decomposed;

    if (!PyArg_ParseTuple(args,
                          "OOOOOi",
                          &handle_obj,
                          &A_obj,
                          &S_obj,
                          &C_obj,
                          &eps_obj,
                          &is_already_decomposed))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);

    int err;
    double *ev = (double *)PyArray_DATA(eps_obj);
    double *a = (double *)PyArray_DATA(A_obj);
    double *b = (double *)PyArray_DATA(S_obj);
    double *q = (double *)PyArray_DATA(C_obj);

    if(PyArray_DESCR(A_obj)->type_num == NPY_DOUBLE) {
        elpa_generalized_eigenvectors(handle, a, b, ev, q,
                                      is_already_decomposed, &err);

    } else {
        elpa_generalized_eigenvectors(handle,
                                      (double complex *)a,
                                      (double complex *)b,
                                      ev,
                                      (double complex *)q,
                                      is_already_decomposed, &err);
    }
    return checkerr(err);
}

PyObject *pyelpa_hermitian_multiply(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    int ncb;

    char *uplo_a, *uplo_c;
    int nrows_b, ncols_b, nrows_c, ncols_c;

    PyArrayObject *A_obj, *B_obj, *C_obj;

    if(!PyArg_ParseTuple(args, "OssiOOiiOii",
                         &handle_obj, &uplo_a, &uplo_c,
                         &ncb, &A_obj, &B_obj, &nrows_b, &ncols_b,
                         &C_obj, &nrows_c, &ncols_c)) {
        return NULL;
    }

    elpa_t handle = unpack_handle(handle_obj);
    int err;

    double *A = (double *)PyArray_DATA(A_obj);
    double *B = (double *)PyArray_DATA(B_obj);
    double *C = (double *)PyArray_DATA(C_obj);

    // The elpa_hermitian_multiply() has a typo, which means it will
    // resolve as a compilation error in the precompiler.
    // We shall have to call the _d function explicitly then.
    if(PyArray_DESCR(A_obj)->type_num == NPY_DOUBLE) {
        elpa_hermitian_multiply_d(handle, uplo_a[0], uplo_c[0], ncb,
                                  A, B, nrows_b, ncols_b,
                                  C, nrows_c, ncols_c,
                                  &err);
    } else {
        elpa_hermitian_multiply_dc(handle, uplo_a[0], uplo_c[0], ncb,
                                   (double complex *)A,
                                   (double complex *)B,
                                   nrows_b, ncols_b,
                                   (double complex *)C,
                                   nrows_c, ncols_c,
                                   &err);
    }
    return checkerr(err);
}

PyObject *pyelpa_deallocate(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if(!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    elpa_t handle = unpack_handle(handle_obj);
    elpa_deallocate(handle);
    // This function provides no error checking
    Py_RETURN_NONE;
}

#endif
