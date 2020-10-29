#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

Matrix2cd D_inhout(Vector2d, Vector2d, MatrixXd, MatrixXd, MatrixXd, double, double);
VectorXcd Chifold_v2(Vector2d, MatrixXd, Matrix2cd, Matrix2cd, Vector2cd, Vector2cd, Matrix2d);
Vector3d energyf_sp(MatrixXd, double, double, Vector2d, Vector2d, Vector2d, MatrixXd, MatrixXd, MatrixXd, MatrixXd, double, MatrixXd, MatrixXd, MatrixXd, VectorXd, int, int, double, Vector2d, int, MatrixXd, double);
MatrixXd visD_uftot(MatrixXd, double, double, Vector2d, Vector2d, Vector2d, MatrixXd, MatrixXd, MatrixXd, MatrixXd, double, MatrixXd, MatrixXd, MatrixXd, VectorXd, int, int, double, Vector2d, int, MatrixXd, double);
MatrixXd visuftot(MatrixXd, double, double, Vector2d, Vector2d, Vector2d, MatrixXd, MatrixXd, MatrixXd, MatrixXd, double, MatrixXd, MatrixXd, MatrixXd, VectorXd, int, int, double, Vector2d, int, MatrixXd, double);

Matrix2d tensprod4x2(MatrixXd, Matrix2d);
Matrix2cd tensprod4x2c(MatrixXd, Matrix2cd);
MatrixXd tensprodnx1(MatrixXd, Vector2d, int);