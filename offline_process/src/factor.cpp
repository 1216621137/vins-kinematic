#include "factor.h"
double p_cost(Eigen::Vector3d ref, Eigen::Vector3d pi, Eigen::Vector3d pj, Eigen::Quaterniond qi, double w){
    Eigen::Vector3d i_pij = qi.conjugate()*(pj-pi);
    Eigen::Vector3d res = (i_pij - ref)*w;
    return 0.5*res.transpose()*res;
}

double q_cost(Eigen::Quaterniond ref, Eigen::Quaterniond qi, Eigen::Quaterniond qj, double w){
    Eigen::Quaterniond i_qij = qi.conjugate()*qj;
    Eigen::Vector3d res = (ref.conjugate()*i_qij).vec()*2*w;
    return 0.5*res.transpose()*res;
    //return 0;
}

double kinematic_cost(Eigen::Vector3d a, Eigen::Vector3d b, double w){
    Eigen::Vector3d e = (a-b)*w;
    return 0.5*e.transpose()*e;
}


bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}

KinematicPoseFactor::KinematicPoseFactor(const Vector3d &x, const Vector3d &p, const Matrix3d &info)
                                                                                                :xoi(x), Po(p), _info(info){ };
bool KinematicPoseFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians)const{
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Map<Eigen::Vector3d> residual(residuals);
     
    residual = (Pi + Qi*xoi) - Po; 
    //residual =  residual;
    if(jacobians){
        if(jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pi(jacobians[0]);
            Eigen::Matrix<double, 3, 6> jacobian_ppi;
            jacobian_ppi.leftCols<3>() = Matrix3d::Identity();
            jacobian_ppi.rightCols<3>() = Qi.toRotationMatrix() * -Utility::skewSymmetric(xoi);
            //jacobian_ppi = _info *jacobian_ppi;
            jacobian_pi.leftCols<6>() = jacobian_ppi;
            jacobian_pi.rightCols<1>().setZero();
            //ROS_INFO("you are in ceres!");
        }
    }
    return true;
};


KinematicVeloFactor::KinematicVeloFactor(Vector3d x, Vector3d v, Vector3d c_w, Matrix3d info):Vo(v), _info(info){                
                                                                                                                Wo << 0, -c_w(2), c_w(1),
                                                                                                                            c_w(2), 0, -c_w(0),
                                                                                                                            -c_w(1), c_w(0), 0;
                                                                                                                            
                                                                                                                xoi = Wo * x;
                                                                                                                
                                                                                                            }      
bool KinematicVeloFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);

    Eigen::Map<Eigen::Vector3d> residual(residuals);
    residual = (Vi + Qi*xoi) - Vo;
    if(jacobians){
        if(jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pi(jacobians[0]);
            jacobian_pi.setZero();
            jacobian_pi.block<3, 3>(0,3) = Qi.toRotationMatrix() * -Utility::skewSymmetric(xoi);
            //jacobian_ppi = _info *jacobian_ppi;
            //ROS_INFO("you are in ceres!");
        }
        if(jacobians[1]){
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_vi(jacobians[1]);
            jacobian_vi.setZero();
            jacobian_vi.block<3, 3>(0, 0) = Matrix3d::Identity();

            //ROS_INFO("you are in ceres!");
        }
    }
    return true;
};