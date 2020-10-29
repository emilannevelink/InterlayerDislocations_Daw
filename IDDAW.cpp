//_cDSCDAW.cpp

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <math.h>
#include <set>
#include "ID_Daw.h"

using namespace Eigen;
using namespace std;

// Compile Using g++ -std=c++11 IDDAWwd.cpp -o IDDAWtest
// Make sure Main is uncommented
// Based on C from DSCDAW_revised.cpp

const complex<double> I(0, 1);
const double PI = acos(-1);
const double eps = 1e-8;


Matrix2cd D_inhout(Vector2d G, Vector2d Lxy, MatrixXd loc, MatrixXd dir, MatrixXd burgers, VectorXd rc, double pn){
	double G2 = G.dot(G);
	double OmegaA = Lxy.prod();

	Vector2d loci;
	Vector2d diri;
	Vector2d burgersi;
	Vector2d burgersie; 
	Vector2d burgersis;

	int ndis = loc.rows();

	Matrix2cd Distortion_tensori = MatrixXcd::Zero(2,2);
	Matrix2cd Distortion_tensor = MatrixXcd::Zero(2,2);
	

	/*
	cout << "G: "<< G << " G2: "<< G2 << endl;
	cout << "Lxy: " << Lxy << " p_Lxy: " << OmegaA << endl;
	cout << "Locations: "<< endl << loc << endl;
	cout << "burgers: " << endl << burgers << endl;
	*/
	//cout << "Rc: " << endl << rc << endl;
	double dangle;
	double gangle = fmod((atan(G(1)/G(0))+5*PI/2),PI);
	if (G(1)<0){
		gangle += PI;
	}

	for (int j=0; j<ndis; j++){
		loci = loc.row(j);
		diri = dir.row(j);
		burgersi = burgers.row(j);

		if (diri(0) == 0){
			if (diri(1)>0){
				dangle = PI/2;
			}
			else{
				dangle = -PI/2;
			}
		} else {
			dangle = fmod(atan(diri(1)/diri(0))+2*PI,PI);
			if (diri(1)<0){
				dangle+=PI;
			}
		}

		dangle += 2*PI;

		burgersis = burgersi.dot(diri)*diri/diri.dot(diri);
		burgersie = burgersi - burgersis;

		if (abs(fmod(dangle,PI)-fmod(gangle,PI))/abs(fmod(dangle,PI))<eps || G2==0 || (fmod(dangle,PI)-fmod(gangle,PI))==0){
			Distortion_tensori(0,0) = diri(1)*(burgersie(0)*exp(-G2*pow(rc(2*j+1),2)/4.)+burgersis(0)*exp(-G2*pow(rc(2*j),2)/4.));
			Distortion_tensori(1,0) = -diri(0)*(burgersie(0)*exp(-G2*pow(rc(2*j+1),2)/4.)+burgersis(0)*exp(-G2*pow(rc(2*j),2)/4.));
			Distortion_tensori(0,1) = diri(1)*(burgersie(1)*exp(-G2*pow(rc(2*j+1),2)/4.)+burgersis(1)*exp(-G2*pow(rc(2*j),2)/4.));
			Distortion_tensori(1,1) = -diri(0)*(burgersie(1)*exp(-G2*pow(rc(2*j+1),2)/4.)+burgersis(1)*exp(-G2*pow(rc(2*j),2)/4.));
			
			Distortion_tensori*=exp(-I*G.dot(loci));
			//cout << exp(-I*cc*G.dot(loci)) << endl;
			//cout << Distortion_tensori << endl;
			Distortion_tensor += Distortion_tensori;
		}
		

	}
	/*
	cout << exp(-G2*pow(rc,2)/4.) << endl;
	cout << erf(Lxy(0)/rc) << endl;
	cout << OmegaA << endl;
	cout << pn << endl;
	*/
	Distortion_tensor *= pn/OmegaA;
	//cout << Distortion_tensor << endl;

	return Distortion_tensor;

}


VectorXcd Chifold_v2(Vector2d G, MatrixXd Cijkl, Matrix2cd D_inh1, Matrix2cd D_inh2, Vector2cd u1, Vector2cd u2, Matrix2d Ajl){
	double G2 = G.dot(G);

	
	if (G2 == 0){
		VectorXcd ret(4);
		ret(0) = 0; ret(1) = 0;
		ret(2) = 0; ret(3) = 0;
		return ret;
	}
	

	MatrixXd ttt = tensprodnx1(Cijkl,G,1);
	Matrix2d CGG = tensprodnx1(ttt,G,1);

	MatrixXd A(4,4);

	A(0,0) = CGG(0,0) + 2*Ajl(0,0);
	A(0,1) = CGG(0,1) + 2*Ajl(0,1);
	A(0,2) = - 2*Ajl(0,0);
	A(0,3) = - 2*Ajl(0,1);
	A(1,0) = CGG(1,0) + 2*Ajl(1,0);
	A(1,1) = CGG(1,1) + 2*Ajl(1,1);
	A(1,2) = -2*Ajl(1,0);
	A(1,3) = -2*Ajl(1,1);
	A(2,0) = -2*Ajl(0,0);
	A(2,1) = -2*Ajl(0,1);
	A(2,2) = CGG(0,0) + 2*Ajl(0,0);
	A(2,3) = CGG(0,1) + 2*Ajl(0,1);
	A(3,0) = -2*Ajl(1,0);
	A(3,1) = -2*Ajl(1,1);
	A(3,2) = CGG(1,0) + 2*Ajl(1,0);
	A(3,3) = CGG(1,1) + 2*Ajl(1,1);
	

	//cout << "A\n" << A << endl;
	

	VectorXcd BB(4);

	BB(0) = -((tensprod4x2c(Cijkl,D_inh1)*G)(0)+(2.0*I*Ajl*(u1-u2))(0));
	BB(1) = -((tensprod4x2c(Cijkl,D_inh1)*G)(1)+(2.0*I*Ajl*(u1-u2))(1));
	BB(2) = -((tensprod4x2c(Cijkl,D_inh2)*G)(0)-(2.0*I*Ajl*(u1-u2))(0));
	BB(3) = -((tensprod4x2c(Cijkl,D_inh2)*G)(1)-(2.0*I*Ajl*(u1-u2))(1));

	//cout << "BB\n" << BB << endl;

    MatrixXd Ainv = A.inverse();

    //cout << "Ainv\n" << Ainv << endl;

	VectorXcd chij1 = Ainv*BB;

	//cout << "chij\n" << chij1 << endl;

	return chij1;
}


MatrixXd visD_uftot(MatrixXd Cijkl, double kap, double c33, Vector2d Lxy, Vector2d a1, Vector2d a2, MatrixXd fG1, MatrixXd fG2, MatrixXd loc_in, MatrixXd burgers_in, double rc_in, MatrixXd loc_out, MatrixXd dir_out, MatrixXd burgers_out, VectorXd rc_out, int pmax, int qmax, double z0, Vector2d M,int ncoor, MatrixXd X, double alpha){
	/*
	cout << "Cijkl" << endl << Cijkl << endl;
	cout << "kap" << endl << kap << endl;
	cout << "c33" << endl << c33 << endl;
	cout << "Lxy" << endl << Lxy << endl;
	cout << "a1" << endl << a1 << endl;
	cout << "a2" << endl << a2 << endl;
	cout << "fG1" << endl << fG1 << endl;
	cout << "fG2" << endl << fG2 << endl;
	cout << "loc" << endl << loc_in << endl;
	cout << "burgers" << endl << burgers_in << endl;
	cout << "rc" << endl << rc_in << endl;
	cout << "loc_out" << endl << loc_out << endl;
	cout << "dir_out" << endl << dir_out << endl;
	cout << "burgers_out" << endl << burgers_out << endl;
	cout << "rc_out" << endl << rc_out << endl;
	cout << "pmax" << endl << pmax << endl;
	cout << "qmax" << endl << qmax << endl;
	cout << "z0" << endl << z0 << endl;
	cout << "M" << endl << M << endl;
	*/
	
	//cout << "fG1" << endl << fG1 << endl;
	//cout << "fG2" << endl << fG2 << endl;

	//cout << "loc" << endl << loc << endl;

	//cout << "burgers" << endl << burgers << endl;

	//cout << "a1" << endl << a1 << endl;
	//cout << "a2" << endl << a2 << endl;

	Vector2d G;
	double OmegaA = Lxy.prod();

	Matrix2cd D_inter1;
	Matrix2cd D_inter2;

	MatrixXd ux = MatrixXd::Constant(ncoor,4,0);

	for (int p=-pmax; p < pmax+1; p++){
		for (int q = -qmax; q < qmax+1; q++){
			G(0) = p*a2(1)-q*a1(1); G(1)=-p*a2(0)+q*a1(0);
			G *= 2*PI/(OmegaA);
			
			D_inter1 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,1)/2;
			D_inter2 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,-1)/2;
			
			double G2 = G.dot(G);

			for (int i=0; i<ncoor; i++){
				if (G2 == 0){
					ux(i,0) += real((D_inter1(0,0)*X(i,0)+D_inter1(1,0)*X(i,1)));
					ux(i,1) += real((D_inter1(0,1)*X(i,0)+D_inter1(1,1)*X(i,1)));
					ux(i,2) += real((D_inter2(0,0)*X(i,0)+D_inter2(1,0)*X(i,1)));
					ux(i,3) += real((D_inter2(0,1)*X(i,0)+D_inter2(1,1)*X(i,1)));
				} else {
					ux(i,0) += real(-I*(D_inter1(0,0)*G(0)+D_inter1(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,1) += real(-I*(D_inter1(0,1)*G(0)+D_inter1(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,2) += real(-I*(D_inter2(0,0)*G(0)+D_inter2(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,3) += real(-I*(D_inter2(0,1)*G(0)+D_inter2(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				}
			}
		}
	}

	int ndis_out = loc_out.rows();

	Vector2d loci;
	Vector2d diri;
	Vector2d burgersi;

	for (int i=0; i<ncoor; i++){
		for (int j=0; j<ndis_out; j++){
			loci = loc_out.row(j);
			diri = dir_out.row(j);
			burgersi = burgers_out.row(j);
			
			if (diri(0)==0) {
				if (X(i,0)>loci(0)) {
					ux(i,0) -= burgersi(0)/2;
					ux(i,1) -= burgersi(1)/2;
					ux(i,2) += burgersi(0)/2;
					ux(i,3) += burgersi(1)/2;
				}

			} else {
				if ((X(i,1))>(diri(1)/diri(0)*(X(i,0)-loci(0)))+loci(1)) {
					ux(i,0) -= burgersi(0)/2;
					ux(i,1) -= burgersi(1)/2;
					ux(i,2) += burgersi(0)/2;
					ux(i,3) += burgersi(1)/2;
				}
				if (diri(1) !=0 ) {//Dislocation periodic images
					if (((X(i,1)-loci(1)-a1(1))/(diri(1)/diri(0))+loci(0)+a1(0))<X(i,0)){
						ux(i,0) += burgersi(0)/2*copysign(1,diri(1)/diri(0));
						ux(i,1) += burgersi(1)/2*copysign(1,diri(1)/diri(0));
						ux(i,2) -= burgersi(0)/2*copysign(1,diri(1)/diri(0));
						ux(i,3) -= burgersi(1)/2*copysign(1,diri(1)/diri(0));
					}
				}
			}
			

			

		}
	}

	Vector2cd ur_fold1;
	Vector2cd ur_fold2;
	Vector2cd ur_tot1;
	Vector2cd ur_tot2;
	Vector2cd ur_12;

	Matrix2cd Dr_fold1;
	Matrix2cd Dr_fold2;

	Matrix2cd D1_tot;
	Matrix2cd D2_tot;

	Matrix2d Ajl;
	//Construct Stacking Matrix
	Ajl(0,0) = alpha;
	Ajl(0,1) = 0;
	Ajl(1,0) = 0;
	Ajl(1,1) = alpha;

	VectorXcd Chi(4);
	Vector2cd Chi1;
	Vector2cd Chi2;

	double Ee1 = 0;
	double Ee2 = 0;

	double Eint = 0;
	Matrix2cd D_hom1;
	Matrix2cd D_hom2;

	MatrixXd Dx = MatrixXd::Constant(ncoor,8,0);

	for (int p=-pmax; p < pmax+1; p++){
		for (int q = -qmax; q < qmax+1; q++){
			G(0) = p*a2(1)-q*a1(1); G(1)=-p*a2(0)+q*a1(0);
			G *= 2*PI/(OmegaA);
			
			ur_fold1 = Vector2cd::Constant(2,0);
			ur_fold2 = Vector2cd::Constant(2,0);
			
			double G2 = G.dot(G);

			if (G2 > 0){
				for (int i=0; i<ncoor; i++){
					ur_fold1(0) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,0)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold1(1) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,1)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold2(0) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,2)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold2(1) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,3)/(double(2*pmax+1))/(double(2*qmax+1));
				}	
			} 
			

			D_inter1 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,1)/2;
			D_inter2 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,-1)/2;

			Chi = Chifold_v2(G, Cijkl, D_inter1, D_inter2, ur_fold1, ur_fold2, Ajl);
			Chi1(0) = Chi(0); Chi1(1) = Chi(1);
			Chi2(0) = Chi(2); Chi2(1) = Chi(3);

			D_hom1 = G*Chi1.transpose();
			D_hom2 = G*Chi2.transpose();
			
			D1_tot = D_inter1 + D_hom1;
			D2_tot = D_inter2 + D_hom2;

			Ee1 += OmegaA/2*real((D1_tot*tensprod4x2c(Cijkl,D1_tot.adjoint())).trace());
			Ee2 += OmegaA/2*real((D2_tot*tensprod4x2c(Cijkl,D2_tot.adjoint())).trace());

			ur_tot1 = (ur_fold1-I*Chi1);
			ur_tot2 = (ur_fold2-I*Chi2);
			ur_12 = (ur_tot1-ur_tot2);
			
			Eint += OmegaA*real(alpha*ur_12.dot(ur_12.conjugate()));
			
			for (int i=0; i<ncoor; i++){
				Dx(i,0) += real(D1_tot(0,0)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				Dx(i,1) += real(D1_tot(0,1)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				Dx(i,2) += real(D1_tot(1,0)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				Dx(i,3) += real(D1_tot(1,1)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				Dx(i,4) += real(D2_tot(0,0)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				Dx(i,5) += real(D2_tot(0,1)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				Dx(i,6) += real(D2_tot(1,0)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				Dx(i,7) += real(D2_tot(1,1)*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
			}
		}
	}

	//cout << "X" << endl << X << endl;

	//cout << "Dx" << endl << Dx << endl;
	
	return Dx;

}


MatrixXd visuftot(MatrixXd Cijkl, double kap, double c33, Vector2d Lxy, Vector2d a1, Vector2d a2, MatrixXd fG1, MatrixXd fG2, MatrixXd loc_in, MatrixXd burgers_in, double rc_in, MatrixXd loc_out, MatrixXd dir_out, MatrixXd burgers_out, VectorXd rc_out, int pmax, int qmax, double z0, Vector2d M,int ncoor, MatrixXd X, double alpha){
	/*
	cout << "Cijkl" << endl << Cijkl << endl;
	cout << "kap" << endl << kap << endl;
	cout << "c33" << endl << c33 << endl;
	cout << "Lxy" << endl << Lxy << endl;
	cout << "a1" << endl << a1 << endl;
	cout << "a2" << endl << a2 << endl;
	cout << "fG1" << endl << fG1 << endl;
	cout << "fG2" << endl << fG2 << endl;
	cout << "loc" << endl << loc_in << endl;
	cout << "burgers" << endl << burgers_in << endl;
	cout << "rc" << endl << rc_in << endl;
	cout << "loc_out" << endl << loc_out << endl;
	cout << "dir_out" << endl << dir_out << endl;
	cout << "burgers_out" << endl << burgers_out << endl;
	cout << "rc_out" << endl << rc_out << endl;
	cout << "pmax" << endl << pmax << endl;
	cout << "qmax" << endl << qmax << endl;
	cout << "z0" << endl << z0 << endl;
	cout << "M" << endl << M << endl;
	*/
	
	//cout << "fG1" << endl << fG1 << endl;
	//cout << "fG2" << endl << fG2 << endl;

	//cout << "loc" << endl << loc << endl;

	//cout << "burgers" << endl << burgers << endl;

	//cout << "a1" << endl << a1 << endl;
	//cout << "a2" << endl << a2 << endl;

	Vector2d G;
	double OmegaA = Lxy.prod();

	Matrix2cd D_inter1;
	Matrix2cd D_inter2;

	MatrixXd ux = MatrixXd::Constant(ncoor,4,0);

	
	for (int p=-pmax; p < pmax+1; p++){
		for (int q = -qmax; q < qmax+1; q++){
			G(0) = p*a2(1)-q*a1(1); G(1)=-p*a2(0)+q*a1(0);
			G *= 2*PI/(OmegaA);
			
			D_inter1 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,1)/2;
			D_inter2 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,-1)/2;
			
			double G2 = G.dot(G);

			for (int i=0; i<ncoor; i++){
				if (G2 == 0){
					ux(i,0) += real((D_inter1(0,0)*X(i,0)+D_inter1(1,0)*X(i,1)));
					ux(i,1) += real((D_inter1(0,1)*X(i,0)+D_inter1(1,1)*X(i,1)));
					ux(i,2) += real((D_inter2(0,0)*X(i,0)+D_inter2(1,0)*X(i,1)));
					ux(i,3) += real((D_inter2(0,1)*X(i,0)+D_inter2(1,1)*X(i,1)));
				} else {
					ux(i,0) += real(-I*(D_inter1(0,0)*G(0)+D_inter1(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,1) += real(-I*(D_inter1(0,1)*G(0)+D_inter1(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,2) += real(-I*(D_inter2(0,0)*G(0)+D_inter2(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,3) += real(-I*(D_inter2(0,1)*G(0)+D_inter2(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				}
			}
		}
	}

	int ndis_out = loc_out.rows();

	Vector2d loci;
	Vector2d diri;
	Vector2d burgersi;

	for (int i=0; i<ncoor; i++){
		for (int j=0; j<ndis_out; j++){
			loci = loc_out.row(j);
			diri = dir_out.row(j);
			burgersi = burgers_out.row(j);
			
			if (diri(0)==0) {
				if (X(i,0)>loci(0)) {
					ux(i,0) -= burgersi(0)/2;
					ux(i,1) -= burgersi(1)/2;
					ux(i,2) += burgersi(0)/2;
					ux(i,3) += burgersi(1)/2;
				}

			} else {
				if ((X(i,1))>(diri(1)/diri(0)*(X(i,0)-loci(0)))+loci(1)) {
					ux(i,0) -= burgersi(0)/2;
					ux(i,1) -= burgersi(1)/2;
					ux(i,2) += burgersi(0)/2;
					ux(i,3) += burgersi(1)/2;
				}
				if (diri(1) !=0 ) {//Dislocation periodic images
					if (((X(i,1)-loci(1)-a1(1))/(diri(1)/diri(0))+loci(0)+a1(0))<X(i,0)){
						ux(i,0) += burgersi(0)/2*copysign(1,diri(1)/diri(0));
						ux(i,1) += burgersi(1)/2*copysign(1,diri(1)/diri(0));
						ux(i,2) -= burgersi(0)/2*copysign(1,diri(1)/diri(0));
						ux(i,3) -= burgersi(1)/2*copysign(1,diri(1)/diri(0));
					}
				}
			}
			

			

		}
	}

	Vector2cd ur_fold1;
	Vector2cd ur_fold2;
	Vector2cd ur_tot1;
	Vector2cd ur_tot2;
	Vector2cd ur_12;

	Matrix2cd Dr_fold1;
	Matrix2cd Dr_fold2;

	Matrix2cd D1_tot;
	Matrix2cd D2_tot;

	Matrix2d Ajl;
	//Construct Stacking Matrix
	Ajl(0,0) = alpha;
	Ajl(0,1) = 0;
	Ajl(1,0) = 0;
	Ajl(1,1) = alpha;

	VectorXcd Chi(4);
	Vector2cd Chi1;
	Vector2cd Chi2;

	double Ee1 = 0;
	double Ee2 = 0;

	double Eint = 0;
	Matrix2cd D_hom1;
	Matrix2cd D_hom2;

	MatrixXd utot = MatrixXd::Constant(ncoor,4,0);

	for (int p=-pmax; p < pmax+1; p++){
		for (int q = -qmax; q < qmax+1; q++){
			G(0) = p*a2(1)-q*a1(1); G(1)=-p*a2(0)+q*a1(0);
			G *= 2*PI/(OmegaA);
			
			ur_fold1 = Vector2cd::Constant(2,0);
			ur_fold2 = Vector2cd::Constant(2,0);
			
			double G2 = G.dot(G);

			if (G2 > 0){
				for (int i=0; i<ncoor; i++){
					ur_fold1(0) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,0)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold1(1) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,1)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold2(0) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,2)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold2(1) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,3)/(double(2*pmax+1))/(double(2*qmax+1));
				}	
			} 
			

			D_inter1 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,1)/2;
			D_inter2 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,-1)/2;

			Chi = Chifold_v2(G, Cijkl, D_inter1, D_inter2, ur_fold1, ur_fold2, Ajl);
			Chi1(0) = Chi(0); Chi1(1) = Chi(1);
			Chi2(0) = Chi(2); Chi2(1) = Chi(3);

			D_hom1 = G*Chi1.transpose();
			D_hom2 = G*Chi2.transpose();
			
			D1_tot = D_inter1 + D_hom1;
			D2_tot = D_inter2 + D_hom2;

			Ee1 += OmegaA/2*real((D1_tot*tensprod4x2c(Cijkl,D1_tot.adjoint())).trace());
			Ee2 += OmegaA/2*real((D2_tot*tensprod4x2c(Cijkl,D2_tot.adjoint())).trace());

			ur_tot1 = (ur_fold1-I*Chi1);
			ur_tot2 = (ur_fold2-I*Chi2);
			ur_12 = (ur_tot1-ur_tot2);
			ur_12 = ((I*ur_fold1+Chi1)-(I*ur_fold2+Chi2));
			
			Eint += OmegaA*real(alpha*ur_12.dot(ur_12.conjugate()));
			
			for (int i=0; i<ncoor; i++){
				if (G2 == 0){
					utot(i,0) += real((D1_tot(0,0)*X(i,0)+D1_tot(1,0)*X(i,1)));
					utot(i,1) += real((D1_tot(0,1)*X(i,0)+D1_tot(1,1)*X(i,1)));
					utot(i,2) += real((D2_tot(0,0)*X(i,0)+D2_tot(1,0)*X(i,1)));
					utot(i,3) += real((D2_tot(0,1)*X(i,0)+D2_tot(1,1)*X(i,1)));
				} else {
					utot(i,0) += real(-I*(D1_tot(0,0)*G(0)+D1_tot(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					utot(i,1) += real(-I*(D1_tot(0,1)*G(0)+D1_tot(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					utot(i,2) += real(-I*(D2_tot(0,0)*G(0)+D2_tot(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					utot(i,3) += real(-I*(D2_tot(0,1)*G(0)+D2_tot(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				}
			}

		}
	}

	cout << utot << endl;
	cout << "visualize ufold_tot" << endl;
	cout << rc_out << endl;

	return utot;

}


Vector3d energyf_sp(MatrixXd Cijkl, double kap, double c33, Vector2d Lxy, Vector2d a1, Vector2d a2, MatrixXd fG1, MatrixXd fG2, MatrixXd loc_in, MatrixXd burgers_in, double rc_in, MatrixXd loc_out, MatrixXd dir_out, MatrixXd burgers_out, VectorXd rc_out, int pmax, int qmax, double z0, Vector2d M,int ncoor, MatrixXd X, double alpha){
	/*
	cout << "Cijkl" << endl << Cijkl << endl;
	cout << "kap" << endl << kap << endl;
	cout << "c33" << endl << c33 << endl;
	cout << "Lxy" << endl << Lxy << endl;
	cout << "a1" << endl << a1 << endl;
	cout << "a2" << endl << a2 << endl;
	cout << "fG1" << endl << fG1 << endl;
	cout << "fG2" << endl << fG2 << endl;
	cout << "loc" << endl << loc_in << endl;
	cout << "burgers" << endl << burgers_in << endl;
	cout << "rc" << endl << rc_in << endl;
	cout << "loc_out" << endl << loc_out << endl;
	cout << "dir_out" << endl << dir_out << endl;
	cout << "burgers_out" << endl << burgers_out << endl;
	cout << "rc_out" << endl << rc_out << endl;
	cout << "pmax" << endl << pmax << endl;
	cout << "qmax" << endl << qmax << endl;
	cout << "z0" << endl << z0 << endl;
	cout << "M" << endl << M << endl;
	*/
	

	Vector2d G;
	double OmegaA = Lxy.prod();

	Matrix2cd D_inter1;
	Matrix2cd D_inter2;

	MatrixXd ux = MatrixXd::Constant(ncoor,4,0);

	for (int p=-pmax; p < pmax+1; p++){
		for (int q = -qmax; q < qmax+1; q++){
			G(0) = p*a2(1)-q*a1(1); G(1)=-p*a2(0)+q*a1(0);
			G *= 2*PI/(OmegaA);
			
			D_inter1 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,1)/2;
			D_inter2 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,-1)/2;

			double G2 = G.dot(G);

			for (int i=0; i<ncoor; i++){
				if (G2 == 0){
					ux(i,0) += real((D_inter1(0,0)*X(i,0)+D_inter1(1,0)*X(i,1)));
					ux(i,1) += real((D_inter1(0,1)*X(i,0)+D_inter1(1,1)*X(i,1)));
					ux(i,2) += real((D_inter2(0,0)*X(i,0)+D_inter2(1,0)*X(i,1)));
					ux(i,3) += real((D_inter2(0,1)*X(i,0)+D_inter2(1,1)*X(i,1)));
				} else {
					ux(i,0) += real(-I*(D_inter1(0,0)*G(0)+D_inter1(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,1) += real(-I*(D_inter1(0,1)*G(0)+D_inter1(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,2) += real(-I*(D_inter2(0,0)*G(0)+D_inter2(1,0)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
					ux(i,3) += real(-I*(D_inter2(0,1)*G(0)+D_inter2(1,1)*G(1))/G2*exp(I*(G(0)*X(i,0)+G(1)*X(i,1))));
				}
			}
		}
	}

	int ndis_out = loc_out.rows();

	Vector2d loci;
	Vector2d diri;
	Vector2d burgersi;

	for (int i=0; i<ncoor; i++){
		for (int j=0; j<ndis_out; j++){
			loci = loc_out.row(j);
			diri = dir_out.row(j);
			burgersi = burgers_out.row(j);
			
			if (diri(0)==0) {
				if (X(i,0)>loci(0)) {
					ux(i,0) -= burgersi(0)/2;
					ux(i,1) -= burgersi(1)/2;
					ux(i,2) += burgersi(0)/2;
					ux(i,3) += burgersi(1)/2;
				}

			} else {
				if ((X(i,1))>(diri(1)/diri(0)*(X(i,0)-loci(0)))+loci(1)) {
					ux(i,0) -= burgersi(0)/2;
					ux(i,1) -= burgersi(1)/2;
					ux(i,2) += burgersi(0)/2;
					ux(i,3) += burgersi(1)/2;
				}
				if (diri(1) !=0 ) {//Dislocation periodic images
					if (((X(i,1)-loci(1)-a1(1))/(diri(1)/diri(0))+loci(0)+a1(0))<X(i,0)){
						ux(i,0) += burgersi(0)/2*copysign(1,diri(1)/diri(0));
						ux(i,1) += burgersi(1)/2*copysign(1,diri(1)/diri(0));
						ux(i,2) -= burgersi(0)/2*copysign(1,diri(1)/diri(0));
						ux(i,3) -= burgersi(1)/2*copysign(1,diri(1)/diri(0));
					}
				}
			}
			

			

		}
	}

	Vector2cd ur_fold1;
	Vector2cd ur_fold2;
	Vector2cd ur_12;
	Vector2cd ur_12star;

	Matrix2cd Dr_fold1;
	Matrix2cd Dr_fold2;

	Matrix2cd D1_tot;
	Matrix2cd D2_tot;

	Matrix2d Ajl;
	//Construct Stacking Matrix
	Ajl(0,0) = alpha;
	Ajl(0,1) = 0;
	Ajl(1,0) = 0;
	Ajl(1,1) = alpha;

	VectorXcd Chi(4);
	Vector2cd Chi1;
	Vector2cd Chi2;

	double Ee1 = 0;
	double Ee2 = 0;

	double Eint = 0;
	Matrix2cd D_hom1;
	Matrix2cd D_hom2;

	for (int p=-pmax; p < pmax+1; p++){
		for (int q = -qmax; q < qmax+1; q++){
			G(0) = p*a2(1)-q*a1(1); G(1)=-p*a2(0)+q*a1(0);
			G *= 2*PI/(OmegaA);
			
			ur_fold1 = Vector2cd::Zero();
			ur_fold2 = Vector2cd::Zero();
			
			double G2 = G.dot(G);

			if (G2 > 0){
				for (int i=0; i<ncoor; i++){
					ur_fold1(0) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,0)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold1(1) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,1)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold2(0) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,2)/(double(2*pmax+1))/(double(2*qmax+1));
					ur_fold2(1) += exp(-I*(G(0)*X(i,0)+G(1)*X(i,1)))*ux(i,3)/(double(2*pmax+1))/(double(2*qmax+1));
				}	
			} 

			D_inter1 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,1)/2;
			D_inter2 = D_inhout(G,Lxy,loc_out,dir_out,burgers_out,rc_out,-1)/2;

			Chi = Chifold_v2(G, Cijkl, D_inter1, D_inter2, ur_fold1, ur_fold2, Ajl);
			Chi1(0) = Chi(0); Chi1(1) = Chi(1);
			Chi2(0) = Chi(2); Chi2(1) = Chi(3);


			D_hom1 = G*Chi1.transpose();
			D_hom2 = G*Chi2.transpose();
			
			D1_tot = D_inter1 + D_hom1;
			D2_tot = D_inter2 + D_hom2;
			
			Ee1 += OmegaA/2*real((D1_tot*tensprod4x2c(Cijkl,D1_tot.adjoint())).trace());
			Ee2 += OmegaA/2*real((D2_tot*tensprod4x2c(Cijkl,D2_tot.adjoint())).trace());

			ur_12 = ((I*ur_fold1+Chi1)-(I*ur_fold2+Chi2));
			
			ur_12star = ur_12.conjugate();
			if (G2>0){
				Eint += OmegaA*real(alpha*(ur_12(0)*ur_12star(0)+ur_12(1)*ur_12star(1)));	
			}
			
			/*
			cout << "G" << G << endl;
			cout << "E1" << OmegaA/2*real((D1_tot*tensprod4x2c(Cijkl,D1_tot.adjoint())).trace()) << endl;
			cout << "E2" << OmegaA/2*real((D2_tot*tensprod4x2c(Cijkl,D2_tot.adjoint())).trace()) << endl;
			cout << "Eint" << OmegaA*real(alpha*(ur_12(0)*ur_12star(0)+ur_12(1)*ur_12star(1))) << endl;
			*/
			
		}
	}

	Vector3d E_split(Ee1, Ee2, Eint);
	return E_split;


}

/*
int main(){
	//cout << "test0" <<endl;
	Vector2d Lxy(100.0, 100);
	Vector2d a1(100.0, 0);
	Vector2d a2(0, 100);

	double OmegaA = Lxy.prod();

	MatrixXd loc_in(0,2);
	
	//loc_in(0,0) = 88.8097724915;
	//loc_in(0,1) = 4.98933043207;
	//loc_in(1,0) = 88.8048248291;
	//loc_in(1,1) = 15.5353250504;
	

	MatrixXd burgers_in(0,2);
	
	//burgers_in(0,0) = -1.7187100851e-05;
	//burgers_in(0,1) = 0.554614810946;
	//burgers_in(1,0) = 1.7187100851e-05;
	//burgers_in(1,1) = -0.554614810946;
	

	double rc_in = 0.5488299236083054;
	
	MatrixXd loc_out(1,2);
	
	loc_out(0,0) = Lxy(0)/2;
	loc_out(0,1) = 0;

	MatrixXd dir_out(1,2);
	
	dir_out(0,0) = 0;
	dir_out(0,1) = 1;
	

	MatrixXd burgers_out(1,2);

	burgers_out(0,0) = 10;
	burgers_out(0,1) = 0;
	

	double rc_out = 5;

	//cout << "test1" <<endl;
	MatrixXd C(2,8); //Make sure to set zero values to zero.
	C(0,0) = 19.528647315; C(0,1) = 0; C(0,2) = 0; C(0,3) = 6.9; C(0,4) = 0; C(0,5) = 6.9; C(0,6) = 5.72864721; C(0,7) = 0;
	C(1,0) = 0; C(1,1) = 5.72864721; C(1,2) = 6.9; C(1,3) = 0; C(1,4) = 6.9; C(1,5) = 0; C(1,6) = 0; C(1,7) = 19.52864721;

	
	int p = 1; int q = 1;

	Vector2d G = {p*a2(1)-q*a1(1),-p*a2(0)+q*a1(0)};
	G *= PI/(OmegaA/2);

	//Matrix2cd D_inh = Distortion_inh(G,Lxy,loc,burgers,rc_in,1);
	//Matrix2cd D_inhcc = Distortion_inh(G,Lxy,loc,burgers,rc_in,-1);

	//cout << G << endl << D_inh << endl << D_inhcc << endl;

	//Vector2d energyi = energy(C, Lxy, a1, a2, loc, burgers, rc_in, tol,Emax);

	//cout << "Test Function: " <<endl << energyi << endl;

	//D_inh = Distortion_inh(G,Lxy,loc,burgers,rc_in,1)*I;
	//D_inhcc = Distortion_inh(G,Lxy,loc,burgers,rc_in,-1)*I;

	//cout << G << endl << D_inh << endl << D_inhcc << endl;

	//double enn = findEnergy(D_inh, C, D_inhcc, G, OmegaA);

	//cout << enn << endl;
	
	double z0 = 3.4;
	int pmax = 1;
	int qmax = 0;
	double kap = 1.38;
	double c33 = 5.382;
	Vector2d M;
	M(0) = 0; M(1) = 0;

	MatrixXd fG1 = MatrixXd::Constant(pmax*2+1,qmax*2+1,1);
	fG1(pmax+1,qmax) = 0;
	fG1(pmax,qmax) = z0;
	MatrixXd fG2 = MatrixXd::Constant(pmax*2+1,qmax*2+1,1);

	cout << "test" <<endl;
	double energyi = energy(C, kap, c33, Lxy, a1, a2, fG1, fG2, loc_in, burgers_in, rc_in, loc_out, dir_out, burgers_out, rc_out, pmax, qmax, z0, M);

	VectorXd der = ederivative(C, kap, c33, Lxy, a1, a2, fG1, fG2, loc_in, burgers_in, rc_in, loc_out, dir_out, burgers_out, rc_out, pmax, qmax, z0, M);


	cout << "Test Function: " << endl << energyi << endl;

	cout << "Test Derivative: " << endl << der << endl;

}
*/

MatrixXd tensprodnx1(MatrixXd C, Vector2d V, int ind){
    
    int ncols = C.size()/2;
    MatrixXd out(2,ncols/2);
    
    if (ncols == 8){
        if (ind == 1){
            out.block(0,0,2,2)=C.block(0,0,2,2)*V(0);
            out.block(0,0,2,2)+=C.block(0,2,2,2)*V(1);
            out.block(0,2,2,2)=C.block(0,4,2,2)*V(0);
            out.block(0,2,2,2)+=C.block(0,6,2,2)*V(1);
        }
    }
    if (ncols == 4){
        Vector2d temp;
        if (ind == 1){
            temp = C.block(0,0,2,2)*V;
            //cout << temp << endl;
            out.block(0,0,1,2)=temp.transpose();
            temp = C.block(0,2,2,2)*V;
            //cout << temp << endl;
            out.block(1,0,1,2)=temp.transpose();
        }
    }

    //cout << C << endl << V << endl << out << endl;
    return out;
}


Matrix2d tensprod4x2(MatrixXd C, Matrix2d D){
	//Shape of C is 2x8; Shape of D is 2x2
	Matrix2d out;
	
	out(0,0) = (C.block(0,0,2,2)*D).trace();
	out(0,1) = (C.block(0,2,2,2)*D).trace();
	out(1,0) = (C.block(0,4,2,2)*D).trace();
	out(1,1) = (C.block(0,6,2,2)*D).trace();

	return out;
}

Matrix2cd tensprod4x2c(MatrixXd C, Matrix2cd D){
	//Shape of C is 2x8; Shape of D is 2x2
	Matrix2cd out;
	
	out(0,0) = (C.block(0,0,2,2)*D).trace();
	out(0,1) = (C.block(0,2,2,2)*D).trace();
	out(1,0) = (C.block(0,4,2,2)*D).trace();
	out(1,1) = (C.block(0,6,2,2)*D).trace();

	return out;
}