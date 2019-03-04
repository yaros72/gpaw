/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2007-2010  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#include <numpy/arrayobject.h>
#ifdef PARALLEL
#include <mpi.h>
#endif
#include <xc.h>

#define PY3 (PY_MAJOR_VERSION >= 3)

#ifdef GPAW_HPM
PyObject* ibm_hpm_start(PyObject *self, PyObject *args);
PyObject* ibm_hpm_stop(PyObject *self, PyObject *args);
PyObject* ibm_mpi_start(PyObject *self);
PyObject* ibm_mpi_stop(PyObject *self);
#endif

#ifdef CRAYPAT
#include <pat_api.h>
PyObject* craypat_region_begin(PyObject *self, PyObject *args);
PyObject* craypat_region_end(PyObject *self, PyObject *args);
#endif

PyObject* symmetrize(PyObject *self, PyObject *args);
PyObject* symmetrize_ft(PyObject *self, PyObject *args);
PyObject* symmetrize_wavefunction(PyObject *self, PyObject *args);
PyObject* symmetrize_return_index(PyObject *self, PyObject *args);
PyObject* symmetrize_with_index(PyObject *self, PyObject *args);
PyObject* map_k_points(PyObject *self, PyObject *args);
PyObject* scal(PyObject *self, PyObject *args);
PyObject* mmm(PyObject *self, PyObject *args);
PyObject* tetrahedron_weight(PyObject *self, PyObject *args);
PyObject* gemm(PyObject *self, PyObject *args);
PyObject* gemv(PyObject *self, PyObject *args);
PyObject* axpy(PyObject *self, PyObject *args);
PyObject* czher(PyObject *self, PyObject *args);
PyObject* rk(PyObject *self, PyObject *args);
PyObject* r2k(PyObject *self, PyObject *args);
PyObject* dotc(PyObject *self, PyObject *args);
PyObject* dotu(PyObject *self, PyObject *args);
PyObject* multi_dotu(PyObject *self, PyObject *args);
PyObject* multi_axpy(PyObject *self, PyObject *args);
PyObject* diagonalize(PyObject *self, PyObject *args);
PyObject* diagonalize_mr3(PyObject *self, PyObject *args);
PyObject* general_diagonalize(PyObject *self, PyObject *args);
PyObject* inverse_cholesky(PyObject *self, PyObject *args);
PyObject* banded_cholesky(PyObject* self, PyObject* args);
PyObject* solve_banded_cholesky(PyObject* self, PyObject* args);
PyObject* inverse_symmetric(PyObject *self, PyObject *args);
PyObject* inverse_general(PyObject *self, PyObject *args);
PyObject* linear_solve_band(PyObject *self, PyObject *args);
PyObject* linear_solve_tridiag(PyObject *self, PyObject *args);
PyObject* right_eigenvectors(PyObject *self, PyObject *args);
PyObject* NewLocalizedFunctionsObject(PyObject *self, PyObject *args);
PyObject* NewOperatorObject(PyObject *self, PyObject *args);
PyObject* NewWOperatorObject(PyObject *self, PyObject *args);
PyObject* NewSplineObject(PyObject *self, PyObject *args);
PyObject* NewTransformerObject(PyObject *self, PyObject *args);
PyObject* pc_potential(PyObject *self, PyObject *args);
PyObject* add_to_density(PyObject *self, PyObject *args);
PyObject* utilities_gaussian_wave(PyObject *self, PyObject *args);
PyObject* utilities_vdot(PyObject *self, PyObject *args);
PyObject* utilities_vdot_self(PyObject *self, PyObject *args);
PyObject* errorfunction(PyObject *self, PyObject *args);
PyObject* cerf(PyObject *self, PyObject *args);
PyObject* pack(PyObject *self, PyObject *args);
PyObject* unpack(PyObject *self, PyObject *args);
PyObject* unpack_complex(PyObject *self, PyObject *args);
PyObject* hartree(PyObject *self, PyObject *args);
PyObject* localize(PyObject *self, PyObject *args);
PyObject* NewXCFunctionalObject(PyObject *self, PyObject *args);
PyObject* NewlxcXCFunctionalObject(PyObject *self, PyObject *args);
PyObject* lxcXCFuncNum(PyObject *self, PyObject *args);
PyObject* exterior_electron_density_region(PyObject *self, PyObject *args);
PyObject* plane_wave_grid(PyObject *self, PyObject *args);
PyObject* tci_overlap(PyObject *self, PyObject *args);
PyObject *pwlfc_expand(PyObject *self, PyObject *args);
PyObject *pw_insert(PyObject *self, PyObject *args);
PyObject *pw_precond(PyObject *self, PyObject *args);
PyObject* overlap(PyObject *self, PyObject *args);
PyObject* vdw(PyObject *self, PyObject *args);
PyObject* vdw2(PyObject *self, PyObject *args);
PyObject* spherical_harmonics(PyObject *self, PyObject *args);
PyObject* spline_to_grid(PyObject *self, PyObject *args);
PyObject* NewLFCObject(PyObject *self, PyObject *args);
PyObject* globally_broadcast_bytes(PyObject *self, PyObject *args);
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
PyObject* new_blacs_context(PyObject *self, PyObject *args);
PyObject* get_blacs_gridinfo(PyObject* self, PyObject *args);
PyObject* get_blacs_local_shape(PyObject* self, PyObject *args);
PyObject* blacs_destroy(PyObject *self, PyObject *args);
PyObject* scalapack_set(PyObject *self, PyObject *args);
PyObject* scalapack_redist(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_MR3
PyObject* scalapack_diagonalize_mr3(PyObject *self, PyObject *args);
#endif
PyObject* scalapack_general_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_general_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_MR3
PyObject* scalapack_general_diagonalize_mr3(PyObject *self, PyObject *args);
#endif
PyObject* scalapack_inverse_cholesky(PyObject *self, PyObject *args);
PyObject* scalapack_inverse(PyObject *self, PyObject *args);
PyObject* scalapack_solve(PyObject *self, PyObject *args);
PyObject* pblas_tran(PyObject *self, PyObject *args);
PyObject* pblas_gemm(PyObject *self, PyObject *args);
PyObject* pblas_hemm(PyObject *self, PyObject *args);
PyObject* pblas_gemv(PyObject *self, PyObject *args);
PyObject* pblas_r2k(PyObject *self, PyObject *args);
PyObject* pblas_rk(PyObject *self, PyObject *args);
#if defined(GPAW_WITH_ELPA)
#include <elpa/elpa.h>
PyObject* pyelpa_allocate(PyObject *self, PyObject *args);
PyObject* pyelpa_set(PyObject *self, PyObject *args);
PyObject* pyelpa_set_comm(PyObject *self, PyObject *args);
PyObject* pyelpa_setup(PyObject *self, PyObject *args);
PyObject* pyelpa_diagonalize(PyObject *self, PyObject *args);
PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args);
PyObject* pyelpa_hermitian_multiply(PyObject *self, PyObject *args);
PyObject* pyelpa_constants(PyObject *self, PyObject *args);
PyObject* pyelpa_deallocate(PyObject *self, PyObject *args);
#endif // GPAW_WITH_ELPA
#endif // GPAW_WITH_SL and PARALLEL

#ifdef GPAW_PAPI
PyObject* papi_mem_info(PyObject *self, PyObject *args);
#endif

#ifdef GPAW_WITH_LIBVDWXC
PyObject* libvdwxc_create(PyObject *self, PyObject *args);
PyObject* libvdwxc_has(PyObject* self, PyObject *args);
PyObject* libvdwxc_init_serial(PyObject *self, PyObject *args);
PyObject* libvdwxc_calculate(PyObject *self, PyObject *args);
PyObject* libvdwxc_tostring(PyObject *self, PyObject *args);
PyObject* libvdwxc_free(PyObject* self, PyObject* args);
PyObject* libvdwxc_init_mpi(PyObject* self, PyObject* args);
PyObject* libvdwxc_init_pfft(PyObject* self, PyObject* args);
#endif // GPAW_WITH_LIBVDWXC

#ifdef GPAW_GITHASH
// For converting contents of a macro to a string, see
// https://en.wikipedia.org/wiki/C_preprocessor#Token_stringification
#define STR(s) #s
#define XSTR(s) STR(s)
PyObject* githash(PyObject* self, PyObject* args)
{
    return Py_BuildValue("s", XSTR(GPAW_GITHASH));
}
#undef XSTR
#undef STR
#endif // GPAW_GITHASH

// Moving least squares interpolation
PyObject* mlsqr(PyObject *self, PyObject *args);

static PyMethodDef functions[] = {
    {"symmetrize", symmetrize, METH_VARARGS, 0},
    {"symmetrize_ft", symmetrize_ft, METH_VARARGS, 0},
    {"symmetrize_wavefunction", symmetrize_wavefunction, METH_VARARGS, 0},
    {"symmetrize_return_index", symmetrize_return_index, METH_VARARGS, 0},
    {"symmetrize_with_index", symmetrize_with_index, METH_VARARGS, 0},
    {"map_k_points", map_k_points, METH_VARARGS, 0},
    {"scal", scal, METH_VARARGS, 0},
    {"mmm", mmm, METH_VARARGS, 0},
    {"tetrahedron_weight", tetrahedron_weight, METH_VARARGS, 0},
    {"gemm", gemm, METH_VARARGS, 0},
    {"gemv", gemv, METH_VARARGS, 0},
    {"axpy", axpy, METH_VARARGS, 0},
    {"czher", czher, METH_VARARGS, 0},
    {"rk",  rk,  METH_VARARGS, 0},
    {"r2k", r2k, METH_VARARGS, 0},
    {"dotc", dotc, METH_VARARGS, 0},
    {"dotu", dotu, METH_VARARGS, 0},
    {"multi_dotu", multi_dotu, METH_VARARGS, 0},
    {"multi_axpy", multi_axpy, METH_VARARGS, 0},
    {"diagonalize", diagonalize, METH_VARARGS, 0},
    {"diagonalize_mr3", diagonalize_mr3, METH_VARARGS, 0},
    {"general_diagonalize", general_diagonalize, METH_VARARGS, 0},
    {"inverse_cholesky", inverse_cholesky, METH_VARARGS, 0},
    {"banded_cholesky", banded_cholesky, METH_VARARGS, 0},
    {"solve_banded_cholesky", solve_banded_cholesky, METH_VARARGS, 0},
    {"inverse_symmetric", inverse_symmetric, METH_VARARGS, 0},
    {"inverse_general", inverse_general, METH_VARARGS, 0},
    {"linear_solve_band", linear_solve_band, METH_VARARGS, 0},
    {"linear_solve_tridiag", linear_solve_tridiag, METH_VARARGS, 0},
    {"right_eigenvectors", right_eigenvectors, METH_VARARGS, 0},
    {"LocalizedFunctions", NewLocalizedFunctionsObject, METH_VARARGS, 0},
    {"Operator", NewOperatorObject, METH_VARARGS, 0},
    {"WOperator", NewWOperatorObject, METH_VARARGS, 0},
    {"Spline", NewSplineObject, METH_VARARGS, 0},
    {"Transformer", NewTransformerObject, METH_VARARGS, 0},
    {"add_to_density", add_to_density, METH_VARARGS, 0},
    {"utilities_gaussian_wave", utilities_gaussian_wave, METH_VARARGS, 0},
    {"utilities_vdot", utilities_vdot, METH_VARARGS, 0},
    {"utilities_vdot_self", utilities_vdot_self, METH_VARARGS, 0},
    {"eed_region", exterior_electron_density_region, METH_VARARGS, 0},
    {"plane_wave_grid", plane_wave_grid, METH_VARARGS, 0},
    {"pwlfc_expand", pwlfc_expand, METH_VARARGS, 0},
    {"pw_insert", pw_insert, METH_VARARGS, 0},
    {"pw_precond", pw_precond, METH_VARARGS, 0},
    {"erf", errorfunction, METH_VARARGS, 0},
    {"cerf", cerf, METH_VARARGS, 0},
    {"pack", pack, METH_VARARGS, 0},
    {"unpack", unpack, METH_VARARGS, 0},
    {"unpack_complex", unpack_complex,           METH_VARARGS, 0},
    {"hartree", hartree, METH_VARARGS, 0},
    {"localize", localize, METH_VARARGS, 0},
    {"XCFunctional", NewXCFunctionalObject, METH_VARARGS, 0},
    {"lxcXCFunctional", NewlxcXCFunctionalObject, METH_VARARGS, 0},
    {"lxcXCFuncNum", lxcXCFuncNum, METH_VARARGS, 0},
    {"overlap", overlap, METH_VARARGS, 0},
    {"tci_overlap", tci_overlap, METH_VARARGS, 0},
    {"vdw", vdw, METH_VARARGS, 0},
    {"vdw2", vdw2, METH_VARARGS, 0},
    {"spherical_harmonics", spherical_harmonics, METH_VARARGS, 0},
    {"pc_potential", pc_potential, METH_VARARGS, 0},
    {"spline_to_grid", spline_to_grid, METH_VARARGS, 0},
    {"LFC", NewLFCObject, METH_VARARGS, 0},
    {"globally_broadcast_bytes", globally_broadcast_bytes, METH_VARARGS, 0},
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
    {"new_blacs_context", new_blacs_context, METH_VARARGS, NULL},
    {"get_blacs_gridinfo", get_blacs_gridinfo, METH_VARARGS, NULL},
    {"get_blacs_local_shape", get_blacs_local_shape, METH_VARARGS, NULL},
    {"blacs_destroy", blacs_destroy, METH_VARARGS, 0},
    {"scalapack_set", scalapack_set, METH_VARARGS, 0},
    {"scalapack_redist", scalapack_redist, METH_VARARGS, 0},
    {"scalapack_diagonalize_dc", scalapack_diagonalize_dc, METH_VARARGS, 0},
    {"scalapack_diagonalize_ex", scalapack_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
    {"scalapack_diagonalize_mr3", scalapack_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
    {"scalapack_general_diagonalize_dc",
     scalapack_general_diagonalize_dc, METH_VARARGS, 0},
    {"scalapack_general_diagonalize_ex",
     scalapack_general_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
    {"scalapack_general_diagonalize_mr3",
     scalapack_general_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
    {"scalapack_inverse_cholesky", scalapack_inverse_cholesky,
     METH_VARARGS, 0},
    {"scalapack_inverse", scalapack_inverse, METH_VARARGS, 0},
    {"scalapack_solve", scalapack_solve, METH_VARARGS, 0},
    {"pblas_tran", pblas_tran, METH_VARARGS, 0},
    {"pblas_gemm", pblas_gemm, METH_VARARGS, 0},
    {"pblas_hemm", pblas_hemm, METH_VARARGS, 0},
    {"pblas_gemv", pblas_gemv, METH_VARARGS, 0},
    {"pblas_r2k", pblas_r2k, METH_VARARGS, 0},
    {"pblas_rk", pblas_rk, METH_VARARGS, 0},
#if defined(GPAW_WITH_ELPA)
    {"pyelpa_allocate", pyelpa_allocate, METH_VARARGS, 0},
    {"pyelpa_set", pyelpa_set, METH_VARARGS, 0},
    {"pyelpa_setup", pyelpa_setup, METH_VARARGS, 0},
    {"pyelpa_set_comm", pyelpa_set_comm, METH_VARARGS, 0},
    {"pyelpa_diagonalize", pyelpa_diagonalize, METH_VARARGS, 0},
    {"pyelpa_general_diagonalize", pyelpa_general_diagonalize, METH_VARARGS, 0},
    {"pyelpa_hermitian_multiply", pyelpa_hermitian_multiply, METH_VARARGS, 0},
    {"pyelpa_constants", pyelpa_constants, METH_VARARGS, 0},
    {"pyelpa_deallocate", pyelpa_deallocate, METH_VARARGS, 0},
#endif // GPAW_WITH_ELPA
#endif // GPAW_WITH_SL && PARALLEL
#ifdef GPAW_HPM
    {"hpm_start", ibm_hpm_start, METH_VARARGS, 0},
    {"hpm_stop", ibm_hpm_stop, METH_VARARGS, 0},
    {"mpi_start", (PyCFunction) ibm_mpi_start, METH_NOARGS, 0},
    {"mpi_stop", (PyCFunction) ibm_mpi_stop, METH_NOARGS, 0},
#endif // GPAW_HPM
#ifdef CRAYPAT
    {"craypat_region_begin", craypat_region_begin, METH_VARARGS, 0},
    {"craypat_region_end", craypat_region_end, METH_VARARGS, 0},
#endif // CRAYPAT
#ifdef GPAW_PAPI
    {"papi_mem_info", papi_mem_info, METH_VARARGS, 0},
#endif // GPAW_PAPI
#ifdef GPAW_WITH_LIBVDWXC
    {"libvdwxc_create", libvdwxc_create, METH_VARARGS, 0},
    {"libvdwxc_has", libvdwxc_has, METH_VARARGS, 0},
    {"libvdwxc_init_serial", libvdwxc_init_serial, METH_VARARGS, 0},
    {"libvdwxc_calculate", libvdwxc_calculate, METH_VARARGS, 0},
    {"libvdwxc_tostring", libvdwxc_tostring, METH_VARARGS, 0},
    {"libvdwxc_free", libvdwxc_free, METH_VARARGS, 0},
    {"libvdwxc_init_mpi", libvdwxc_init_mpi, METH_VARARGS, 0},
    {"libvdwxc_init_pfft", libvdwxc_init_pfft, METH_VARARGS, 0},
#endif // GPAW_WITH_LIBVDWXC
    {"mlsqr", mlsqr, METH_VARARGS, 0},
#ifdef GPAW_GITHASH
    {"githash", githash, METH_VARARGS, 0},
#endif // GPAW_GITHASH
    {0, 0, 0, 0}
};

#ifdef PARALLEL
extern PyTypeObject MPIType;
extern PyTypeObject GPAW_MPI_Request_type;
#endif

extern PyTypeObject LFCType;
extern PyTypeObject LocalizedFunctionsType;
extern PyTypeObject OperatorType;
extern PyTypeObject WOperatorType;
extern PyTypeObject SplineType;
extern PyTypeObject TransformerType;
extern PyTypeObject XCFunctionalType;
extern PyTypeObject lxcXCFunctionalType;

PyObject* globally_broadcast_bytes(PyObject *self, PyObject *args)
{
    PyObject *pybytes;
    if(!PyArg_ParseTuple(args, "O", &pybytes)){
        return NULL;
    }

#ifdef PARALLEL
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    long size;
    if(rank == 0) {
        size = PyBytes_Size(pybytes);  // Py_ssize_t --> long
    }
    MPI_Bcast(&size, 1, MPI_LONG, 0, comm);

    char *dst = (char *)malloc(size);
    if(rank == 0) {
        char *src = PyBytes_AsString(pybytes);  // Read-only
        memcpy(dst, src, size);
    }
    MPI_Bcast(dst, size, MPI_BYTE, 0, comm);

    PyObject *value = PyBytes_FromStringAndSize(dst, size);
    free(dst);
    return value;
#else
    return pybytes;
#endif
}


#if PY3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_gpaw",
    "C-extension for GPAW",
    -1,
    functions,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

static PyObject* moduleinit(void)
{
#ifdef PARALLEL
    if (PyType_Ready(&MPIType) < 0)
        return NULL;
    if (PyType_Ready(&GPAW_MPI_Request_type) < 0)
        return NULL;
#endif

    if (PyType_Ready(&LFCType) < 0)
        return NULL;
    if (PyType_Ready(&LocalizedFunctionsType) < 0)
        return NULL;
    if (PyType_Ready(&OperatorType) < 0)
        return NULL;
    if (PyType_Ready(&WOperatorType) < 0)
        return NULL;
    if (PyType_Ready(&SplineType) < 0)
        return NULL;
    if (PyType_Ready(&TransformerType) < 0)
        return NULL;
    if (PyType_Ready(&XCFunctionalType) < 0)
        return NULL;
    if (PyType_Ready(&lxcXCFunctionalType) < 0)
        return NULL;

#if PY3
    PyObject* m = PyModule_Create(&moduledef);
#else
    PyObject* m = Py_InitModule3("_gpaw", functions,
                                 "C-extension for GPAW\n\n...\n");
#endif

    if (m == NULL)
        return NULL;

#ifdef PARALLEL
    Py_INCREF(&MPIType);
    Py_INCREF(&GPAW_MPI_Request_type);
    PyModule_AddObject(m, "Communicator", (PyObject *)&MPIType);
#endif

    PyObject_SetAttrString(m,
                           "libxc_version",
                           PyUnicode_FromString(xc_version_string()));

    Py_INCREF(&LFCType);
    Py_INCREF(&LocalizedFunctionsType);
    Py_INCREF(&OperatorType);
    Py_INCREF(&WOperatorType);
    Py_INCREF(&SplineType);
    Py_INCREF(&TransformerType);
    Py_INCREF(&XCFunctionalType);
    Py_INCREF(&lxcXCFunctionalType);
#ifndef PARALLEL
    // gpaw-python needs to import arrays at the right time, so this is
    // done in gpaw_main().  In serial, we just do it here:
    import_array1(0);
#endif
    return m;
}

#ifndef GPAW_INTERPRETER


#if PY3
PyMODINIT_FUNC PyInit__gpaw(void)
{
    return moduleinit();
}
#else
PyMODINIT_FUNC init_gpaw(void)
{
    moduleinit();
}
#endif

#else // ifndef GPAW_INTERPRETER

#if PY3
#define moduleinit0 moduleinit
#else
void moduleinit0(void) { moduleinit(); }
#endif


int
gpaw_main()
{
    int status = -1;

    PyObject *gpaw_mod = NULL, *pymain = NULL;

    gpaw_mod = PyImport_ImportModule("gpaw");
    if(gpaw_mod == NULL) {
        status = 3;  // Basic import failure
    } else {
        pymain = PyObject_GetAttrString(gpaw_mod, "main");
    }

    if(pymain == NULL) {
        status = 4;  // gpaw.main does not exist for some reason
        //PyErr_Print();
    } else {
        // Returns Py_None or NULL (error after calling user script)
        // We already imported the Python parts of numpy.  If we want, we can
        // later attempt to broadcast the numpy C API imports, too.
        // However I don't know how many files they are, and we need to
        // figure out how to broadcast extension modules (shared objects).
        import_array1(0);
        PyObject *pyreturn = PyObject_CallFunction(pymain, "");
        status = (pyreturn == NULL);
        Py_XDECREF(pyreturn);
    }

    Py_XDECREF(pymain);
    Py_XDECREF(gpaw_mod);
    return status;
}


int
main(int argc, char **argv)
{
#ifndef GPAW_OMP
    MPI_Init(&argc, &argv);
#else
    int granted;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &granted);
    if (granted != MPI_THREAD_MULTIPLE)
        exit(1);
#endif // GPAW_OMP

#if PY3
#define PyChar wchar_t
    wchar_t* wargv[argc];
    wchar_t* wargv2[argc];
    for (int i = 0; i < argc; i++) {
        int n = 1 + mbstowcs(NULL, argv[i], 0);
        wargv[i] = (wchar_t*)malloc(n * sizeof(wchar_t));
        wargv2[i] = wargv[i];
        mbstowcs(wargv[i], argv[i], n);
    }
#else
#define PyChar char
    char** wargv = argv;
#endif

    Py_SetProgramName(wargv[0]);
    PyImport_AppendInittab("_gpaw", &moduleinit0);
    Py_Initialize();
    PySys_SetArgvEx(argc, wargv, 0);

#ifdef GPAW_WITH_ELPA
    // Globally initialize Elpa library if present:
    if (elpa_init(20171201) != ELPA_OK) {
        // What API versions do we support?
        PyErr_SetString(PyExc_RuntimeError, "Elpa >= 20171201 required");
        PyErr_Print();
        return 1;
    }
#endif

    int status = gpaw_main();

    if(status != 0) {
        PyErr_Print();
    }

#ifdef GPAW_WITH_ELPA
    elpa_uninit();
#endif

    Py_Finalize();
    MPI_Finalize();

#if PY3
    for (int i = 0; i < argc; i++)
        free(wargv2[i]);
#endif

    return status;
}
#endif // GPAW_INTERPRETER
