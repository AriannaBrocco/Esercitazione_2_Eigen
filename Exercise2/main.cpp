#include<iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

bool SystemSolution(const Matrix2d& A,
                    const Vector2d& b,
                    double& determinanteA,
                    double& condizionamentoA,
                    double& erroreRelativo_PALU,
                    double& erroreRelativo_QR,
                    Vector2d& x_PALU,
                    Vector2d& x_QR)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d valoriSingolariA = svd.singularValues();
    condizionamentoA = valoriSingolariA.maxCoeff() / valoriSingolariA.minCoeff();

    determinanteA = A.determinant();

    if( valoriSingolariA.minCoeff() < 1e-16)
    {
        erroreRelativo_PALU = -1;
        erroreRelativo_QR = -1;
        return false;
    }

    Vector2d xesatta;
    xesatta << -1.0e+00, -1.0e+00;

    x_PALU = A.fullPivLu().solve(b);
    x_QR = A.colPivHouseholderQr().solve(b);
    erroreRelativo_PALU = (xesatta - x_PALU).norm() / xesatta.norm();
    erroreRelativo_QR = (xesatta - x_QR).norm() / xesatta.norm();
    return true;
}

int main()
{
    // Definisco la matrice A1 e il vettore b1 per il sistema lineare
    Matrix2d A1;
    Vector2d b1;

    // La soluzione esatta è la stessa per tutti e tre i sistemi
    Vector2d x_esatta;
    x_esatta << -1.0e+00, -1.0e+00;

    // Inizializzo matrice A1 e vettore b1
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    // Dopo aver verificato che la matrice non sia singolare tramite SystemSolution,
    // calcolo la soluzione con fattorizzazione PALU e QR e i relatvi errori
    Vector2d x_PALU1, x_QR1;
    double detA1, condA1, errRelPALU1, errRelQR1;
    if(SystemSolution(A1, b1, detA1, condA1, errRelPALU1, errRelQR1, x_PALU1, x_QR1))
        cout<< scientific<< "Matrice A1 - Determinante di A1: "<< detA1<< ", condizionamento di A1: "<< 1.0 / condA1 << ", errore relativo del primo sistema con Palu: "<< errRelPALU1<< ", errore relativo del primo sistema con QR: "<< errRelQR1 << "\n"
             << "La soluzione ottenuta con la fattorizzazione PALU e': " << x_PALU1 << ".\n" << "La soluzione ottenuta con la fattorizzazione QR e': " << x_QR1 << ".\n" << "La soluzione esatta e': " << x_esatta << "." << endl;

    else
        cout << scientific<< "Matrice A1 - Determinante di A1: "<< detA1<< ", Condizionamento di A1: "<< 1.0 / condA1 << " (La matrice A1 è singolare)"<< endl;

    // Stesso procedimento per gli altri due sistemi
    Matrix2d A2;
    Vector2d b2;

    A2 << 5.547001962252291e-01,-5.540607316466765e-01, 8.320502943378437e-01,-8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;


    Vector2d x_PALU2, x_QR2;
    double detA2, condA2, errRelPALU2, errRelQR2;
    if(SystemSolution(A2, b2, detA2, condA2, errRelPALU2, errRelQR2, x_PALU2, x_QR2))
        cout<< scientific<< "Matrice A2 - Determinante di A2: "<< detA2<< ", condizionamento di A2: "<< 1.0 / condA2 << ", errore relativo del secondo sistema con Palu: "<< errRelPALU2<< ", errore relativo del secondo sistema con QR: "<< errRelQR2 << "\n"
             << "La soluzione ottenuta con la fattorizzazione PALU e': " << x_PALU2 << ".\n" << "La soluzione ottenuta con la fattorizzazione QR e': " << x_QR2 << ".\n" << "La soluzione esatta e': " << x_esatta << "." << endl;

    else
        cout << scientific<< "Matrice A2 - Determinante di A2: "<< detA2<< ", Condizionamento di A2: "<< 1.0 / condA2 << " (La matrice A2 è singolare)"<< endl;


    Matrix2d A3;
    Vector2d b3;

    A3 << 5.547001962252291e-01,-5.547001955851905e-01, 8.320502943378437e-01,-8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    Vector2d x_PALU3, x_QR3;
    double detA3, condA3, errRelPALU3, errRelQR3;
    if(SystemSolution(A3, b3, detA3, condA3, errRelPALU3, errRelQR3, x_PALU3, x_QR3))
        cout<< scientific<< "Matrice A3 - Determinante di A3: "<< detA3<< ", condizionamento di A3: "<< 1.0 / condA3 << ", errore relativo del terzo sistema con Palu: "<< errRelPALU3<< ", errore relativo del terzo sistema con QR: "<< errRelQR3 << "\n"
             << "La soluzione ottenuta con la fattorizzazione PALU e': " << x_PALU3 << ".\n" << "La soluzione ottenuta con la fattorizzazione QR e': " << x_QR3 << ".\n" << "La soluzione esatta e': " << x_esatta << "." << endl;

    else
        cout << scientific<< "Matrice A3 - Determinante di A3: "<< detA3 << ", Condizionamento di A3: "<< 1.0 / condA3 << " (La matrice A3 è singolare)"<< endl;

    return 0;
}
