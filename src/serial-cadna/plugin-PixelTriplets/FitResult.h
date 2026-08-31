#ifndef RecoPixelVertexing_PixelTrackFitting_interface_FitResult_h
#define RecoPixelVertexing_PixelTrackFitting_interface_FitResult_h

#include <cmath>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cadna.h>

namespace Eigen {

template<> struct NumTraits<double_st>
 : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef double_st Real;
  typedef double_st NonInteger;
  typedef double_st Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template<> struct NumTraits<float_st>
 : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef float_st Real;
  typedef float_st NonInteger;
  typedef float_st Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

// Inform Eigen how to handle mixing standard literals with CADNA
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double_st, double, BinaryOp> {
  typedef double_st ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, double_st, BinaryOp> {
  typedef double_st ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double_st, int, BinaryOp> {
  typedef double_st ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<int, double_st, BinaryOp> {
  typedef double_st ReturnType;
};

} // namespace Eigen

namespace Rfit {

  using Vector2d = Eigen::Vector<double_st, 2>;
  using Vector3d = Eigen::Vector<double_st, 3>;
  using Vector4d = Eigen::Vector<double_st, 4>;
  using Vector5d = Eigen::Matrix<double_st, 5, 1>;
  using Matrix2d = Eigen::Matrix<double_st, 2, 2>;
  using Matrix3d = Eigen::Matrix<double_st, 3, 3>;
  using Matrix4d = Eigen::Matrix<double_st, 4, 4>;
  using Matrix5d = Eigen::Matrix<double_st, 5, 5>;
  using Matrix6d = Eigen::Matrix<double_st, 6, 6>;

  template <int N>
  using Matrix3xNd = Eigen::Matrix<double_st, 3, N>;  // used for inputs hits

  struct circle_fit {
    Vector3d par;  //!< parameter: (X0,Y0,R)
    Matrix3d cov;
    /*!< covariance matrix: \n
      |cov(X0,X0)|cov(Y0,X0)|cov( R,X0)| \n
      |cov(X0,Y0)|cov(Y0,Y0)|cov( R,Y0)| \n
      |cov(X0, R)|cov(Y0, R)|cov( R, R)|
    */
    int32_t q;  //!< particle charge
    float_st chi2;
  };

  struct line_fit {
    Vector2d par;  //!<(cotan(theta),Zip)
    Matrix2d cov;
    /*!<
      |cov(c_t,c_t)|cov(Zip,c_t)| \n
      |cov(c_t,Zip)|cov(Zip,Zip)|
    */
    double_st chi2;
  };

  struct helix_fit {
    Vector5d par;  //!<(phi,Tip,pt,cotan(theta)),Zip)
    Matrix5d cov;
    /*!< ()->cov() \n
      |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| \n
      |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| \n
      |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| \n
      |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| \n
      |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
    */
    float_st chi2_circle;
    float_st chi2_line;
    //    Vector4d fast_fit;
    int32_t q;  //!< particle charge
  };            // __attribute__((aligned(16)));

}  // namespace Rfit
#endif
