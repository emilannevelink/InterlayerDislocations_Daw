#include <Python.h>
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <numpy/arrayobject.h>
#include "ID_Daw.h"

/*
This module uses IDDAW.cpp to find the energy of a set of interlayer dislocations
It can be built using setup_ID_DAW.py
and tested using test_IDDAW.py

https://dfm.io/posts/python-c-extensions/
*/


static char module_docstring[] =
    "This module provides an interface for calculating the Continuum interlayer energy.";
static char Energyf_sp_docstring[] =
    "Calculate the Energy of a set of interlayer Dislocations using Daw's formalism.";
static char visDuftot_docstring[] =
    "Calculate the Distortion field of a set of interlayer Dislocations using Daw's formalism.";
static char visuftot_docstring[] =
    "Calculate the in-plane displacement field of a set of interlayer Dislocations using Daw's formalism with stacking.";


static PyObject *IDDAW_Energyf_sp(PyObject *self, PyObject *args);
static PyObject *IDDAW_visDuftot(PyObject *self, PyObject *args);
static PyObject *IDDAW_visuftot(PyObject *self, PyObject *args);

MatrixXd unpacklocbur(double* , int);

static PyMethodDef module_methods[] = {
    {"_Energyf_sp", IDDAW_Energyf_sp, METH_VARARGS, Energyf_sp_docstring},
    {"_visDuftot", IDDAW_visDuftot, METH_VARARGS, visDuftot_docstring},
    {"_visuftot", IDDAW_visuftot, METH_VARARGS, visuftot_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_IDDAW(void)
{
    PyObject *m = Py_InitModule3("_IDDAW", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


static PyObject *IDDAW_Energyf_sp(PyObject *self, PyObject *args)
{
    double rc, kap, c33, z0, alpha;
    int ndisl, ndislout, pmax, qmax, ncoor;
    PyObject *Cijkl_obj, *Lxy_obj, *a1_obj, *a2_obj, *loc_obj, *burgers_obj, *fG1_obj, *fG2_obj, *locout_obj, *dirout_obj, *burgersout_obj, *rcout_obj, *M_obj, *X_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOOOdiOOiiOOOOidddOiOd", &Cijkl_obj, &Lxy_obj, &a1_obj, &a2_obj,
        &loc_obj, &burgers_obj, &rc, &ndisl, &fG1_obj, &fG2_obj, &pmax, &qmax,
        &locout_obj, &dirout_obj, &burgersout_obj, &rcout_obj, &ndislout, &kap, &c33, &z0, &M_obj, &ncoor, &X_obj, &alpha))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *Cijkl_array = PyArray_FROM_OTF(Cijkl_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Lxy_array = PyArray_FROM_OTF(Lxy_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *a1_array = PyArray_FROM_OTF(a1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *a2_array = PyArray_FROM_OTF(a2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *loc_array = PyArray_FROM_OTF(loc_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *burgers_array = PyArray_FROM_OTF(burgers_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *fG1_array = PyArray_FROM_OTF(fG1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *fG2_array = PyArray_FROM_OTF(fG2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *locout_array = PyArray_FROM_OTF(locout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *dirout_array = PyArray_FROM_OTF(dirout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *burgersout_array = PyArray_FROM_OTF(burgersout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *rcout_array = PyArray_FROM_OTF(rcout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *M_array = PyArray_FROM_OTF(M_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (Cijkl_array == NULL || Lxy_array == NULL || a1_array == NULL ||
        a2_array == NULL || loc_array == NULL || burgers_array == NULL ||
        fG1_array == NULL || fG2_array == NULL || locout_array == NULL ||
        burgersout_array == NULL || dirout_array == NULL || M_array == NULL || X_array == NULL) {
        Py_XDECREF(Cijkl_array);
        Py_XDECREF(Lxy_array);
        Py_XDECREF(a1_array);
        Py_XDECREF(a2_array);
        Py_XDECREF(loc_array);
        Py_XDECREF(burgers_array);
        Py_XDECREF(fG1_array);
        Py_XDECREF(fG2_array);
        Py_XDECREF(locout_array);
        Py_XDECREF(dirout_array);
        Py_XDECREF(burgersout_array);
        Py_XDECREF(rcout_array);
        Py_XDECREF(M_array);
        Py_XDECREF(X_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *Cijklp    = (double*)PyArray_DATA(Cijkl_array);
    double *Lxyp    = (double*)PyArray_DATA(Lxy_array);
    double *a1p = (double*)PyArray_DATA(a1_array);
    double *a2p = (double*)PyArray_DATA(a2_array);
    double *locp = (double*)PyArray_DATA(loc_array);
    double *burgersp = (double*)PyArray_DATA(burgers_array);
    double *fG1p = (double*)PyArray_DATA(fG1_array);
    double *fG2p = (double*)PyArray_DATA(fG2_array);
    double *locoutp = (double*)PyArray_DATA(locout_array);
    double *diroutp = (double*)PyArray_DATA(dirout_array);
    double *burgersoutp = (double*)PyArray_DATA(burgersout_array);
    double *rcoutp = (double*)PyArray_DATA(rcout_array);
    double *Mp = (double*)PyArray_DATA(M_array);
    double *Xp = (double*)PyArray_DATA(X_array);

    cout << "pmax" << endl << pmax << endl;
    cout << "qmax" << endl << qmax << endl;

    //Convert Pointers into C Data Types
    MatrixXd C(2,8);
    C(0,0) = Cijklp[0]; C(0,1) = Cijklp[1]; C(0,2) = Cijklp[2]; C(0,3) = Cijklp[3];
    C(0,4) = Cijklp[4]; C(0,5) = Cijklp[5]; C(0,6) = Cijklp[6]; C(0,7) = Cijklp[7];
    C(1,0) = Cijklp[8]; C(1,1) = Cijklp[9]; C(1,2) = Cijklp[10]; C(1,3) = Cijklp[11];
    C(1,4) = Cijklp[12]; C(1,5) = Cijklp[13]; C(1,6) = Cijklp[14]; C(1,7) = Cijklp[15];
    
    MatrixXd burgers(ndisl,2);
    MatrixXd loc(ndisl,2);

    burgers = unpacklocbur(burgersp,ndisl);
    loc = unpacklocbur(locp,ndisl);

    MatrixXd burgersout(ndislout,2);
    MatrixXd locout(ndislout,2);
    MatrixXd dirout(ndislout,2);

    burgersout = unpacklocbur(burgersoutp,ndislout);
    locout = unpacklocbur(locoutp,ndislout);
    dirout = unpacklocbur(diroutp,ndislout);
    VectorXd rcout(2*ndislout);
    for (int k = 0; k<2*ndislout; k++){ 
        rcout(k) = rcoutp[k];
    }

    Vector2d Lxy(Lxyp[0],Lxyp[1]);
    Vector2d a1(a1p[0],a1p[1]);
    Vector2d a2(a2p[0],a2p[1]);

    Vector2d M(Mp[0],Mp[1]);

    MatrixXd fG1 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(fG1p,2*pmax+1,2*qmax+1);
    MatrixXd fG2 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(fG2p,2*pmax+1,2*qmax+1);


    MatrixXd X = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(Xp,ncoor,2);

    /*
    cout << "Cijkl" << endl << C << endl;
    cout << "kap" << endl << kap << endl;
    cout << "c33" << endl << c33 << endl;
    cout << "Lxy" << endl << Lxy << endl;
    cout << "a1" << endl << a1 << endl;
    cout << "a2" << endl << a2 << endl;
    cout << "fG1" << endl << fG1 << endl;
    cout << "fG2" << endl << fG2 << endl;
    cout << "loc" << endl << loc << endl;
    cout << "burgers" << endl << burgers << endl;
    cout << "rc" << endl << rc << endl;
    cout << "loc_out" << endl << locout << endl;
    cout << "dir_out" << endl << dirout << endl;
    cout << "burgers_out" << endl << burgersout << endl;
    cout << "rc_out" << endl << rcout << endl;
    cout << "pmax" << endl << pmax << endl;
    cout << "qmax" << endl << qmax << endl;
    cout << "z0" << endl << z0 << endl;
    cout << "M" << endl << M << endl;
    */
    Vector3d res = energyf_sp(C, kap, c33, Lxy, a1, a2, fG1, fG2, loc, burgers, rc, locout, dirout, burgersout, rcout, pmax, qmax, z0, M, ncoor, X, alpha);

    //double value = res[0];
    double ret_array[3];
    ret_array[0] = res[0];
    ret_array[1] = res[1];
    ret_array[2] = res[2];



    /* Clean up. */
    Py_XDECREF(Cijkl_array);
    Py_XDECREF(Lxy_array);
    Py_XDECREF(a1_array);
    Py_XDECREF(a2_array);
    Py_XDECREF(loc_array);
    Py_XDECREF(burgers_array);
    Py_XDECREF(fG1_array);
    Py_XDECREF(fG2_array);
    Py_XDECREF(locout_array);
    Py_XDECREF(dirout_array);
    Py_XDECREF(burgersout_array);
    Py_XDECREF(rcout_array);
    Py_XDECREF(M_array);
    Py_XDECREF(X_array);

    npy_intp dims[1] = {3};
    PyObject *ret = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(ret), ret_array, sizeof(ret_array));

    /* Build the output tuple */
    //PyObject *ret = Py_BuildValue("d", value);
    return ret;
}

static PyObject *IDDAW_visDuftot(PyObject *self, PyObject *args)
{
    double rc, kap, c33, z0, alpha;
    int ndisl, ndislout, pmax, qmax, ncoor;
    PyObject *Cijkl_obj, *Lxy_obj, *a1_obj, *a2_obj, *loc_obj, *burgers_obj, *fG1_obj, *fG2_obj, *locout_obj, *dirout_obj, *burgersout_obj, *rcout_obj, *M_obj, *X_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOOOdiOOiiOOOOidddOiOd", &Cijkl_obj, &Lxy_obj, &a1_obj, &a2_obj,
        &loc_obj, &burgers_obj, &rc, &ndisl, &fG1_obj, &fG2_obj, &pmax, &qmax,
        &locout_obj, &dirout_obj, &burgersout_obj, &rcout_obj, &ndislout, &kap, &c33, &z0, &M_obj, &ncoor, &X_obj, &alpha))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *Cijkl_array = PyArray_FROM_OTF(Cijkl_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Lxy_array = PyArray_FROM_OTF(Lxy_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *a1_array = PyArray_FROM_OTF(a1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *a2_array = PyArray_FROM_OTF(a2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *loc_array = PyArray_FROM_OTF(loc_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *burgers_array = PyArray_FROM_OTF(burgers_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *fG1_array = PyArray_FROM_OTF(fG1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *fG2_array = PyArray_FROM_OTF(fG2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *locout_array = PyArray_FROM_OTF(locout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *dirout_array = PyArray_FROM_OTF(dirout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *burgersout_array = PyArray_FROM_OTF(burgersout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *rcout_array = PyArray_FROM_OTF(rcout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *M_array = PyArray_FROM_OTF(M_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (Cijkl_array == NULL || Lxy_array == NULL || a1_array == NULL ||
        a2_array == NULL || loc_array == NULL || burgers_array == NULL ||
        fG1_array == NULL || fG2_array == NULL || locout_array == NULL ||
        burgersout_array == NULL || dirout_array == NULL || M_array == NULL || X_array == NULL) {
        Py_XDECREF(Cijkl_array);
        Py_XDECREF(Lxy_array);
        Py_XDECREF(a1_array);
        Py_XDECREF(a2_array);
        Py_XDECREF(loc_array);
        Py_XDECREF(burgers_array);
        Py_XDECREF(fG1_array);
        Py_XDECREF(fG2_array);
        Py_XDECREF(locout_array);
        Py_XDECREF(dirout_array);
        Py_XDECREF(burgersout_array);
        Py_XDECREF(rcout_array);
        Py_XDECREF(M_array);
        Py_XDECREF(X_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *Cijklp    = (double*)PyArray_DATA(Cijkl_array);
    double *Lxyp    = (double*)PyArray_DATA(Lxy_array);
    double *a1p = (double*)PyArray_DATA(a1_array);
    double *a2p = (double*)PyArray_DATA(a2_array);
    double *locp = (double*)PyArray_DATA(loc_array);
    double *burgersp = (double*)PyArray_DATA(burgers_array);
    double *fG1p = (double*)PyArray_DATA(fG1_array);
    double *fG2p = (double*)PyArray_DATA(fG2_array);
    double *locoutp = (double*)PyArray_DATA(locout_array);
    double *diroutp = (double*)PyArray_DATA(dirout_array);
    double *burgersoutp = (double*)PyArray_DATA(burgersout_array);
    double *rcoutp = (double*)PyArray_DATA(rcout_array);
    double *Mp = (double*)PyArray_DATA(M_array);
    double *Xp = (double*)PyArray_DATA(X_array);

    cout << "pmax" << endl << pmax << endl;
    cout << "qmax" << endl << qmax << endl;

    //Convert Pointers into C Data Types
    MatrixXd C(2,8);
    C(0,0) = Cijklp[0]; C(0,1) = Cijklp[1]; C(0,2) = Cijklp[2]; C(0,3) = Cijklp[3];
    C(0,4) = Cijklp[4]; C(0,5) = Cijklp[5]; C(0,6) = Cijklp[6]; C(0,7) = Cijklp[7];
    C(1,0) = Cijklp[8]; C(1,1) = Cijklp[9]; C(1,2) = Cijklp[10]; C(1,3) = Cijklp[11];
    C(1,4) = Cijklp[12]; C(1,5) = Cijklp[13]; C(1,6) = Cijklp[14]; C(1,7) = Cijklp[15];
    
    MatrixXd burgers(ndisl,2);
    MatrixXd loc(ndisl,2);

    burgers = unpacklocbur(burgersp,ndisl);
    loc = unpacklocbur(locp,ndisl);

    MatrixXd burgersout(ndislout,2);
    MatrixXd locout(ndislout,2);
    MatrixXd dirout(ndislout,2);

    burgersout = unpacklocbur(burgersoutp,ndislout);
    locout = unpacklocbur(locoutp,ndislout);
    dirout = unpacklocbur(diroutp,ndislout);
    VectorXd rcout(2*ndislout);
    for (int k = 0; k<2*ndislout; k++){ 
        rcout(k) = rcoutp[k];
    }

    Vector2d Lxy(Lxyp[0],Lxyp[1]);
    Vector2d a1(a1p[0],a1p[1]);
    Vector2d a2(a2p[0],a2p[1]);

    Vector2d M(Mp[0],Mp[1]);

    MatrixXd fG1 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(fG1p,2*pmax+1,2*qmax+1);
    MatrixXd fG2 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(fG2p,2*pmax+1,2*qmax+1);


    MatrixXd X = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(Xp,ncoor,2);

    /*
    cout << "Cijkl" << endl << C << endl;
    cout << "kap" << endl << kap << endl;
    cout << "c33" << endl << c33 << endl;
    cout << "Lxy" << endl << Lxy << endl;
    cout << "a1" << endl << a1 << endl;
    cout << "a2" << endl << a2 << endl;
    cout << "fG1" << endl << fG1 << endl;
    cout << "fG2" << endl << fG2 << endl;
    cout << "loc" << endl << loc << endl;
    cout << "burgers" << endl << burgers << endl;
    cout << "rc" << endl << rc << endl;
    cout << "loc_out" << endl << locout << endl;
    cout << "dir_out" << endl << dirout << endl;
    cout << "burgers_out" << endl << burgersout << endl;
    cout << "rc_out" << endl << rcout << endl;
    cout << "pmax" << endl << pmax << endl;
    cout << "qmax" << endl << qmax << endl;
    cout << "z0" << endl << z0 << endl;
    cout << "M" << endl << M << endl;
    */
    MatrixXd res = visD_uftot(C, kap, c33, Lxy, a1, a2, fG1, fG2, loc, burgers, rc, locout, dirout, burgersout, rcout, pmax, qmax, z0, M, ncoor, X, alpha);

    //cout << "Return: " << res << endl;
    //double value = res[0];
    
    double ret_array[8*ncoor];

    for (int i = 0; i < ncoor; i++){
        for (int j = 0; j < 8; j++){
            ret_array[8*i+j] = res(i,j);
        }
    }



    /* Clean up. */
    Py_XDECREF(Cijkl_array);
    Py_XDECREF(Lxy_array);
    Py_XDECREF(a1_array);
    Py_XDECREF(a2_array);
    Py_XDECREF(loc_array);
    Py_XDECREF(burgers_array);
    Py_XDECREF(fG1_array);
    Py_XDECREF(fG2_array);
    Py_XDECREF(locout_array);
    Py_XDECREF(dirout_array);
    Py_XDECREF(burgersout_array);
    Py_XDECREF(rcout_array);
    Py_XDECREF(M_array);
    Py_XDECREF(X_array);

    //cout << "Return: " << ret_array << endl;

    npy_intp dims[1] = {8*ncoor};
    PyObject *ret = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(ret), ret_array, sizeof(ret_array));

    /* Build the output tuple */
    //PyObject *ret = Py_BuildValue("d", value);
    return ret;
}

static PyObject *IDDAW_visuftot(PyObject *self, PyObject *args)
{
    double rc, kap, c33, z0, alpha;
    int ndisl, ndislout, pmax, qmax, ncoor;
    PyObject *Cijkl_obj, *Lxy_obj, *a1_obj, *a2_obj, *loc_obj, *burgers_obj, *fG1_obj, *fG2_obj, *locout_obj, *dirout_obj, *burgersout_obj, *rcout_obj, *M_obj, *X_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOOOdiOOiiOOOOidddOiOd", &Cijkl_obj, &Lxy_obj, &a1_obj, &a2_obj,
        &loc_obj, &burgers_obj, &rc, &ndisl, &fG1_obj, &fG2_obj, &pmax, &qmax,
        &locout_obj, &dirout_obj, &burgersout_obj, &rcout_obj, &ndislout, &kap, &c33, &z0, &M_obj, &ncoor, &X_obj, &alpha))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *Cijkl_array = PyArray_FROM_OTF(Cijkl_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Lxy_array = PyArray_FROM_OTF(Lxy_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *a1_array = PyArray_FROM_OTF(a1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *a2_array = PyArray_FROM_OTF(a2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *loc_array = PyArray_FROM_OTF(loc_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *burgers_array = PyArray_FROM_OTF(burgers_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *fG1_array = PyArray_FROM_OTF(fG1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *fG2_array = PyArray_FROM_OTF(fG2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *locout_array = PyArray_FROM_OTF(locout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *dirout_array = PyArray_FROM_OTF(dirout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *burgersout_array = PyArray_FROM_OTF(burgersout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *rcout_array = PyArray_FROM_OTF(rcout_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *M_array = PyArray_FROM_OTF(M_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (Cijkl_array == NULL || Lxy_array == NULL || a1_array == NULL ||
        a2_array == NULL || loc_array == NULL || burgers_array == NULL ||
        fG1_array == NULL || fG2_array == NULL || locout_array == NULL ||
        burgersout_array == NULL || dirout_array == NULL || M_array == NULL || X_array == NULL) {
        Py_XDECREF(Cijkl_array);
        Py_XDECREF(Lxy_array);
        Py_XDECREF(a1_array);
        Py_XDECREF(a2_array);
        Py_XDECREF(loc_array);
        Py_XDECREF(burgers_array);
        Py_XDECREF(fG1_array);
        Py_XDECREF(fG2_array);
        Py_XDECREF(locout_array);
        Py_XDECREF(dirout_array);
        Py_XDECREF(burgersout_array);
        Py_XDECREF(rcout_array);
        Py_XDECREF(M_array);
        Py_XDECREF(X_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *Cijklp    = (double*)PyArray_DATA(Cijkl_array);
    double *Lxyp    = (double*)PyArray_DATA(Lxy_array);
    double *a1p = (double*)PyArray_DATA(a1_array);
    double *a2p = (double*)PyArray_DATA(a2_array);
    double *locp = (double*)PyArray_DATA(loc_array);
    double *burgersp = (double*)PyArray_DATA(burgers_array);
    double *fG1p = (double*)PyArray_DATA(fG1_array);
    double *fG2p = (double*)PyArray_DATA(fG2_array);
    double *locoutp = (double*)PyArray_DATA(locout_array);
    double *diroutp = (double*)PyArray_DATA(dirout_array);
    double *burgersoutp = (double*)PyArray_DATA(burgersout_array);
    double *rcoutp = (double*)PyArray_DATA(rcout_array);
    double *Mp = (double*)PyArray_DATA(M_array);
    double *Xp = (double*)PyArray_DATA(X_array);

    cout << "pmax" << endl << pmax << endl;
    cout << "qmax" << endl << qmax << endl;

    //Convert Pointers into C Data Types
    MatrixXd C(2,8);
    C(0,0) = Cijklp[0]; C(0,1) = Cijklp[1]; C(0,2) = Cijklp[2]; C(0,3) = Cijklp[3];
    C(0,4) = Cijklp[4]; C(0,5) = Cijklp[5]; C(0,6) = Cijklp[6]; C(0,7) = Cijklp[7];
    C(1,0) = Cijklp[8]; C(1,1) = Cijklp[9]; C(1,2) = Cijklp[10]; C(1,3) = Cijklp[11];
    C(1,4) = Cijklp[12]; C(1,5) = Cijklp[13]; C(1,6) = Cijklp[14]; C(1,7) = Cijklp[15];
    
    MatrixXd burgers(ndisl,2);
    MatrixXd loc(ndisl,2);

    burgers = unpacklocbur(burgersp,ndisl);
    loc = unpacklocbur(locp,ndisl);

    MatrixXd burgersout(ndislout,2);
    MatrixXd locout(ndislout,2);
    MatrixXd dirout(ndislout,2);

    burgersout = unpacklocbur(burgersoutp,ndislout);
    locout = unpacklocbur(locoutp,ndislout);
    dirout = unpacklocbur(diroutp,ndislout);
    VectorXd rcout(2*ndislout);
    for (int k = 0; k<2*ndislout; k++){ 
        rcout(k) = rcoutp[k];
    }

    Vector2d Lxy(Lxyp[0],Lxyp[1]);
    Vector2d a1(a1p[0],a1p[1]);
    Vector2d a2(a2p[0],a2p[1]);

    Vector2d M(Mp[0],Mp[1]);

    MatrixXd fG1 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(fG1p,2*pmax+1,2*qmax+1);
    MatrixXd fG2 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(fG2p,2*pmax+1,2*qmax+1);


    MatrixXd X = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(Xp,ncoor,2);

    /*
    cout << "Cijkl" << endl << C << endl;
    cout << "kap" << endl << kap << endl;
    cout << "c33" << endl << c33 << endl;
    cout << "Lxy" << endl << Lxy << endl;
    cout << "a1" << endl << a1 << endl;
    cout << "a2" << endl << a2 << endl;
    cout << "fG1" << endl << fG1 << endl;
    cout << "fG2" << endl << fG2 << endl;
    cout << "loc" << endl << loc << endl;
    cout << "burgers" << endl << burgers << endl;
    cout << "rc" << endl << rc << endl;
    cout << "loc_out" << endl << locout << endl;
    cout << "dir_out" << endl << dirout << endl;
    cout << "burgers_out" << endl << burgersout << endl;
    
    cout << "pmax" << endl << pmax << endl;
    cout << "qmax" << endl << qmax << endl;
    cout << "z0" << endl << z0 << endl;
    cout << "M" << endl << M << endl;
    */
    cout << "rc_out" << endl << rcout << endl;
    MatrixXd res = visuftot(C, kap, c33, Lxy, a1, a2, fG1, fG2, loc, burgers, rc, locout, dirout, burgersout, rcout, pmax, qmax, z0, M, ncoor, X, alpha);

    //cout << "Return: " << res << endl;
    //double value = res[0];
    
    double ret_array[4*ncoor];

    for (int i = 0; i < ncoor; i++){
        for (int j = 0; j < 4; j++){
            ret_array[4*i+j] = res(i,j);
        }
    }



    /* Clean up. */
    Py_XDECREF(Cijkl_array);
    Py_XDECREF(Lxy_array);
    Py_XDECREF(a1_array);
    Py_XDECREF(a2_array);
    Py_XDECREF(loc_array);
    Py_XDECREF(burgers_array);
    Py_XDECREF(fG1_array);
    Py_XDECREF(fG2_array);
    Py_XDECREF(locout_array);
    Py_XDECREF(dirout_array);
    Py_XDECREF(burgersout_array);
    Py_XDECREF(rcout_array);
    Py_XDECREF(M_array);
    Py_XDECREF(X_array);

    //cout << "Return: " << ret_array << endl;

    npy_intp dims[1] = {4*ncoor};
    PyObject *ret = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(ret), ret_array, sizeof(ret_array));

    /* Build the output tuple */
    //PyObject *ret = Py_BuildValue("d", value);
    return ret;
}


MatrixXd unpacklocbur(double* point, int ndisl){
	MatrixXd out(ndisl,2);
	for (int i=0;i<ndisl;i++){
		out(i,0) = point[i*2];
		out(i,1) = point[i*2+1];
	}

	return out;
}