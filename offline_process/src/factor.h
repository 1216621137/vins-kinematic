#pragma once
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "../../pose_graph/src/utility/utility.h"
using namespace Eigen;


double p_cost(Eigen::Vector3d ref, Eigen::Vector3d pi, Eigen::Vector3d pj, Eigen::Quaterniond qi, double w);

double q_cost(Eigen::Quaterniond ref, Eigen::Quaterniond qi, Eigen::Quaterniond qj, double w);

double kinematic_cost(Eigen::Vector3d a, Eigen::Vector3d b, double w);
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};
template <typename T>
T NormalizeAngle(const T& angle_degrees) {
  if (angle_degrees > T(180.0))
  	return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
  	return angle_degrees + T(360.0);
  else
  	return angle_degrees;
};
template <typename T> 
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

	T y = yaw / T(180.0) * T(M_PI);
	T p = pitch / T(180.0) * T(M_PI);
	T r = roll / T(180.0) * T(M_PI);


	R[0] = cos(y) * cos(p);
	R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
	R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
	R[3] = sin(y) * cos(p);
	R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
	R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
	R[6] = -sin(p);
	R[7] = cos(p) * sin(r);
	R[8] = cos(p) * cos(r);
};

template <typename T> 
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
	inv_R[0] = R[0];
	inv_R[1] = R[3];
	inv_R[2] = R[6];
	inv_R[3] = R[1];
	inv_R[4] = R[4];
	inv_R[5] = R[7];
	inv_R[6] = R[2];
	inv_R[7] = R[5];
	inv_R[8] = R[8];
};

template <typename T> 
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
	r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
	r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
	r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

template <typename T> inline
void QuaternionMutiply(const T p[4], const T q[4], T result[4]) {
    result[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]; // NOLINT
    result[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2];
    result[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1];
    result[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0];
    const T scale = T(1) / sqrt(result[0] * result[0] +
                              result[1] * result[1] +
                              result[2] * result[2] +
                              result[3] * result[3]);

  // Make unit-norm version of q.
    T norml[4] = {
    scale * result[0],
    scale * result[1],
    scale * result[2],
    scale * result[3],
  };
  result = norml;
}

template <typename T> inline
void LogMap(const T q[4], T result[3]) {
    T q_imag[3] = {q[1], q[2], q[3]};
    T q_imag_squared_norm = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if(q_imag_squared_norm < T(1e-6)){
        result[0] = T(2) * q_imag[0];
        result[1] = T(2) * q_imag[1];
        result[2] = T(2) * q_imag[2];
        return;
    }
    /***
    T q_imag_norm = ceres::sqrt(q_imag_squared_norm);
    T sita = T(2)*ceres::atan2(q_imag_norm, q[0]) / q_imag_norm;//返回角度
    result[0] = sita * q_imag[0];
    result[1] = sita * q_imag[1];
    result[2] = sita * q_imag[2];
    return;
    ***/
}

//连续帧之间的约束
struct sequnce_constrain
{
	sequnce_constrain(Vector3d pij, Quaterniond qij, Vector3d vij,double weight, double weight2, double weight3)
    :_pij(pij), _qij(qij), Weight_p(weight), Weight_q(weight2), Weight_v(weight3), _vij(vij){}

	template <typename T>
	bool operator()(const T* const ti, const T* const tj, const T* const vi, const T* const vj, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];
        T w_R_j[4] = {tj[6], tj[3], tj[4], tj[5]};//wxyz
		T i_R_w[4] = {ti[6], -ti[3], -ti[4], -ti[5]};
        T R_i_j[4];
        T t_i_ij[3];
        ceres::QuaternionRotatePoint(i_R_w, t_w_ij, t_i_ij);
		QuaternionMutiply( i_R_w, w_R_j, R_i_j);
        T measure_q[4] = {T(_qij.w()), T(_qij.x()), T(_qij.y()), T(_qij.z())};//R_ij
        T R_ji[4] ={R_i_j[0], -R_i_j[1], -R_i_j[2], -R_i_j[3]}; 
        T err_q[4];
        QuaternionMutiply( measure_q,R_ji, err_q);
		// 求相对矩阵
		//T err_vector[3];
        /***
        if(err_q[0] < T(0.0)){
            err_q[0] = -err_q[0];
            err_q[1] = -err_q[1];
            err_q[2] = -err_q[2];
            err_q[3] = -err_q[3];
        }
        ***/
        //LogMap(err_q, err_vector);

        T w_vij[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
        T i_v_ij[3];
        ceres::QuaternionRotatePoint(i_R_w, w_vij, i_v_ij);

		residuals[0] = T(Weight_p)*(t_i_ij[0] - T(_pij[0]));
		residuals[1] = T(Weight_p)*(t_i_ij[1] - T(_pij[1]));
		residuals[2] = T(Weight_p)*(t_i_ij[2] - T(_pij[2]));
       
        residuals[3] = T(2) * err_q[1] * T(Weight_q);
        residuals[4] = T(2) * err_q[2]* T(Weight_q);
        residuals[5] = T(2) * err_q[3]* T(Weight_q);
   
        residuals[6] =  T(Weight_v)*(i_v_ij[0] - T(_vij[0]));
        residuals[7] =  T(Weight_v)*(i_v_ij[1] - T(_vij[1]));
        residuals[8] =  T(Weight_v)*(i_v_ij[2] - T(_vij[2]));
        /***
        residuals[3] = T(0);
        residuals[4] = T(0);
        residuals[5] = T(0);
        residuals[6] = T(0);
        residuals[7] = T(0);
        residuals[8] = T(0);
        ***/
		return true;
	}

	static ceres::CostFunction* Create(Vector3d pij, Quaterniond qij, Vector3d vij, double weight, double weight2, double weight3) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          sequnce_constrain, 9, 7, 7, 3, 3>(
	          	new sequnce_constrain(pij, qij, vij, weight, weight2, weight3)));
	}

	Vector3d _pij;
    Quaterniond _qij;
    double Weight_p;
    double Weight_q;
    double Weight_v;
    Vector3d _vij;
};
/***
struct FourDOFError
{
	FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
				  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){}

	template <typename T>
	bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x));
		residuals[1] = (t_i_ij[1] - T(t_y));
		residuals[2] = (t_i_ij[2] - T(t_z));
		residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          FourDOFError, 4, 1, 3, 1, 3>(
	          	new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;

};



class AngleLocalParameterization {
 public:
  template <typename T>

  bool operator()(const T* theta_radians, const T* delta_theta_radians,
                  T* theta_radians_plus_delta) const {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  static ceres::LocalParameterization* Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};

***/










struct KinematicPoseConstrain{
    KinematicPoseConstrain(Vector3d x, Vector3d p, Matrix3d info):
                                                        xoi(x), Po(p), _info(info){ }      
    template <typename T>
    bool operator()( const T* const Posei, T* residuals) const
    {
        //ROS_INFO("you are in 'openrator()'.");
        T x[3] = {T(xoi[0]), T(xoi[1]), T(xoi[2])};
        T x_i[3];
        //qi是子机姿态
        T qi[4] = {Posei[6], Posei[3], Posei[4],Posei[5]};//wxyz
        ceres::QuaternionRotatePoint(qi, x, x_i);
        T mi[3] = {x_i[0]+Posei[0], x_i[1]+Posei[1], x_i[2]+Posei[2]};
        residuals[0] = mi[0] - T(Po[0]);
        residuals[1] = mi[1] - T(Po[1]);
        residuals[2] = mi[2] - T(Po[2]);
        //加上信息矩阵
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
       // ROS_INFO_STREAM("residuals_pose"<<residuals[0]);
        //ROS_INFO("calculation done!!!!!");
        return true;
    }

    static ceres::CostFunction* Create(Vector3d x, Vector3d p, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicPoseConstrain, 3, 7>(
	          	new KinematicPoseConstrain(x, p, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Po;
    Matrix3d _info;
};




struct KinematicVeloConstrain{
    KinematicVeloConstrain(Vector3d x, Vector3d v, Vector3d c_w, Matrix3d info):Vo(v), _info(info){                
        Wo << 0, -c_w(2), c_w(1),
                    c_w(2), 0, -c_w(0),
                    -c_w(1), c_w(0), 0;
        xoi = Wo * x;
    }              
    template <typename T>
    bool operator()(const T* const V, const T* const Posei, T* residuals) const
    {
        
         T x[3] = {T(xoi[0]), T(xoi[1]), T(xoi[2])};
        T x_[3];
        T q[4] = {Posei[6], Posei[3], Posei[4], Posei[5]};
        ceres::QuaternionRotatePoint(q, x, x_);
        T s[3] = {x_[0]+V[0], x_[1]+V[1], x_[2]+V[2]};
        residuals[0] = s[0] - T(Vo[0]);
        residuals[1] = s[1] - T(Vo[1]);
        residuals[2] = s[2] - T(Vo[2]);
        //ROS_INFO_STREAM("residuals_velo"<<residuals[0]);
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain, 3, 3, 7>(
	          	new KinematicVeloConstrain(x, v, w, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Vo;
    Matrix3d Wo;
    Matrix3d _info;
};
/***
struct KinematicPoseConstrain{
    KinematicPoseConstrain(Vector3d x, Vector3d p, Matrix3d info, double Pitch_i, double Roll_i):
                                                        xoi(x), Po(p), _info(info), pitch_i(Pitch_i), roll_i(Roll_i){ }      
    template <typename T>
    bool operator()( const T* const Posei, const T* const yaw_i,  T* residuals) const
    {
        //ROS_INFO("you are in 'openrator()'.");
        T x[3] = {T(xoi[0]), T(xoi[1]), T(xoi[2])};
        T x_i[3];
        //qi是子机姿态
        // euler to rotation
		T qi[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), qi);
        RotationMatrixRotatePoint(qi, x, x_i);
        T mi[3] = {x_i[0]+Posei[0], x_i[1]+Posei[1], x_i[2]+Posei[2]};
        residuals[0] = mi[0] - T(Po[0]);
        residuals[1] = mi[1] - T(Po[1]);
        residuals[2] = mi[2] - T(Po[2]);
        //加上信息矩阵
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
       // ROS_INFO_STREAM("residuals_pose"<<residuals[0]);
        //ROS_INFO("calculation done!!!!!");
        return true;
    }

    static ceres::CostFunction* Create(Vector3d x, Vector3d p, Matrix3d info, double Pitch_i, double Roll_i) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicPoseConstrain, 3, 3, 1>(
	          	new KinematicPoseConstrain(x, p, info, Pitch_i, Roll_i)));
	}

    private:
    Vector3d xoi;
    Vector3d Po;
    Matrix3d _info;
    double pitch_i;
    double roll_i;
};




struct KinematicVeloConstrain{
    KinematicVeloConstrain(Vector3d x, Vector3d v, Vector3d c_w, Matrix3d info, double Pitch_i, double Roll_i)
        :Vo(v), _info(info), pitch_i(Pitch_i), roll_i(Roll_i){                
        Wo << 0, -c_w(2), c_w(1),
                    c_w(2), 0, -c_w(0),
                    -c_w(1), c_w(0), 0;
        xoi = Wo * x;
    }              
    template <typename T>
    bool operator()(const T* const V, const T* const yaw_i, T* residuals) const
    {
        
         T x[3] = {T(xoi[0]), T(xoi[1]), T(xoi[2])};
        T x_[3];
		// euler to rotation
		T q[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), q);
		RotationMatrixRotatePoint(q, x, x_);
        T s[3] = {x_[0]+V[0], x_[1]+V[1], x_[2]+V[2]};
        residuals[0] = s[0] - T(Vo[0]);
        residuals[1] = s[1] - T(Vo[1]);
        residuals[2] = s[2] - T(Vo[2]);
        //ROS_INFO_STREAM("residuals_velo"<<residuals[0]);
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info, double Pitch_i, double Roll_i) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain, 3, 3, 1>(
	          	new KinematicVeloConstrain(x, v, w, info, Pitch_i, Roll_i)));
	}

    private:
    Vector3d xoi;
    Vector3d Vo;
    Matrix3d Wo;
    Matrix3d _info;
    double pitch_i;
    double roll_i;
};
***/

//估计轨迹外参
struct KinematicPoseConstrain_T{
    KinematicPoseConstrain_T(Vector3d x, Vector3d p, Matrix3d info):
                                                        xoi(x), Po(p), _info(info){ }      
    template <typename T>
    bool operator()( const T* const Posei, const T* const _ET, T* residuals) const
    {
        T x[3] = {T(xoi[0]), T(xoi[1]), T(xoi[2])};
        T x_i[3];
        //qi是子机姿态
        T qi[4] = {Posei[6], Posei[3], Posei[4],Posei[5]};//wxyz
        ceres::QuaternionRotatePoint(qi, x, x_i);
        T mi[3] = {x_i[0]+Posei[0], x_i[1]+Posei[1], x_i[2]+Posei[2]};
        //
        T po[3] = {T(Po[0]), T(Po[1]), T(Po[2])};
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, po, x_tr);
        po[0] = x_tr[0]+_ET[0];
        po[1] = x_tr[1]+_ET[1];
        po[2] = x_tr[2]+_ET[2];
        residuals[0] = mi[0] - po[0];
        residuals[1] = mi[1] - po[1];
        residuals[2] = mi[2] - po[2];
        //加上信息矩阵
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
        //ROS_INFO("calculation done!!!!!");
        return true;
    }

    static ceres::CostFunction* Create(Vector3d x, Vector3d p, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicPoseConstrain_T, 3, 7, 7>(
	          	new KinematicPoseConstrain_T(x, p, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Po;
    Matrix3d _info;
};

struct KinematicPoseConstrain_T_x{
    KinematicPoseConstrain_T_x(Vector3d p, Quaterniond q_o, Matrix3d info):
                                                                 Po(p), _q_o(q_o), _info(info){ }      
    template <typename T>
    bool operator()( const T* const Posei, const T* const _ET, const T* const xoi, T* residuals) const
    {
        T _xoi[3] = {xoi[0], xoi[1], xoi[2]};
        T x_i[3];
        //qi是子机姿态
        T qi[4] = {Posei[6], Posei[3], Posei[4],Posei[5]};//wxyz
        ceres::QuaternionRotatePoint(qi, _xoi, x_i);
        T mi[3] = {x_i[0]+Posei[0], x_i[1]+Posei[1], x_i[2]+Posei[2]};
        //
        
        T qo[4] = {T(_q_o.w()), T(_q_o.x()), T(_q_o.y()), T(_q_o.z())};
        T q_delta[3];
        ceres::QuaternionRotatePoint(qo, _xoi, q_delta);
        T po[3] = {T(Po[0])-q_delta[0], T(Po[1])-q_delta[1], T(Po[2])-q_delta[2]};
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, po, x_tr);
        po[0] = x_tr[0]+_ET[0];
        po[1] = x_tr[1]+_ET[1];
        po[2] = x_tr[2]+_ET[2];
        residuals[0] = mi[0] - po[0];
        residuals[1] = mi[1] - po[1];
        residuals[2] = mi[2] - po[2];
        //加上信息矩阵
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
        //ROS_INFO("calculation done!!!!!");
        return true;
    }

    static ceres::CostFunction* Create(Vector3d p, Quaterniond q_o, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicPoseConstrain_T_x, 3, 7, 7, 3>(
	          	new KinematicPoseConstrain_T_x( p, q_o, info)));
	}

    private:
    Vector3d Po;
    Matrix3d _info;
    Quaterniond _q_o;
};


struct KinematicPoseConstrain_T_e2{
    KinematicPoseConstrain_T_e2(Vector3d x, Vector3d p, Matrix3d info):
                                                        xoi(x), Po(p), _info(info){ }      
    template <typename T>
    bool operator()( const T* const Posei, const T* const _ET, T* residuals) const
    {
        T po[3] = {T(Po[0]), T(Po[1]), T(Po[2])};
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, po, x_tr);
        po[0] = x_tr[0]+_ET[0];
        po[1] = x_tr[1]+_ET[1];
        po[2] = x_tr[2]+_ET[2];
        T x[3] = {po[0]-Posei[0], po[1]-Posei[1], po[2]-Posei[2]};
        T i_R_w[4] = {Posei[6], -Posei[3], -Posei[4], -Posei[5]};
        T mi[3];
        ceres::QuaternionRotatePoint(i_R_w, x, mi);
        residuals[0] = mi[0] - T(xoi[0]);
        residuals[1] = mi[1] - T(xoi[1]);
        residuals[2] = mi[2] - T(xoi[2]);
        //加上信息矩阵
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
        //ROS_INFO("calculation done!!!!!");
        return true;
    }

    static ceres::CostFunction* Create(Vector3d x, Vector3d p, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicPoseConstrain_T_e2, 3, 7, 7>(
	          	new KinematicPoseConstrain_T_e2(x, p, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Po;
    Matrix3d _info;
};


struct KinematicVeloConstrain_T{
    KinematicVeloConstrain_T(Vector3d x, Vector3d v, Vector3d c_w, Matrix3d info):Vo(v), _info(info){                
        Wo << 0, -c_w(2), c_w(1),
                    c_w(2), 0, -c_w(0),
                    -c_w(1), c_w(0), 0;
        xoi = Wo * x;
    }              
    template <typename T>
    bool operator()(const T* const V, const T* const Posei, const T* const _ET, T* residuals) const
    {
         T x[3] = {T(xoi[0]), T(xoi[1]), T(xoi[2])};
        T x_[3];
        T q[4] = {Posei[6], Posei[3], Posei[4], Posei[5]};
        ceres::QuaternionRotatePoint(q, x, x_);
        T mi[3] = {x_[0]+V[0], x_[1]+V[1], x_[2]+V[2]};
        //todo
        T vo[3] = {T(Vo[0]), T(Vo[1]), T(Vo[2])};
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, vo, x_tr);
        vo[0] = x_tr[0];
        vo[1] = x_tr[1];
        vo[2] = x_tr[2];
        //todo

        residuals[0] = mi[0] -vo[0];
        residuals[1] = mi[1] - vo[1];
        residuals[2] = mi[2] - vo[2];
        //
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain_T, 3, 3, 7, 7>(
	          	new KinematicVeloConstrain_T(x, v, w, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Vo;
    Matrix3d Wo;
    Matrix3d _info;
};

struct KinematicVeloConstrain_T_x{
    KinematicVeloConstrain_T_x(Vector3d v, Quaterniond Qo, Vector3d c_w, Vector3d c_wo, Matrix3d info):Vo(v), _Qo(Qo), _info(info){                
        Wi << 0, -c_w(2), c_w(1),
                    c_w(2), 0, -c_w(0),
                    -c_w(1), c_w(0), 0;
        Wo << 0, -c_wo(2), c_wo(1),
                        c_wo(2), 0, -c_wo(0),
                        -c_wo(1), c_wo(0), 0;
    }              
    template <typename T>
    bool operator()(const T* const V, const T* const Posei, const T* const _ET,const T* const xoi, T* residuals) const
    {
        T _xoi[3] = {xoi[0], xoi[1], xoi[2]};
        T S_xoi[3];
        S_xoi[0] = T(Wi(0, 0))*_xoi[0] + T(Wi(0, 1))*_xoi[1] + T(Wi(0, 2))*_xoi[2];
        S_xoi[1] = T(Wi(1, 0))*_xoi[0] + T(Wi(1, 1))*_xoi[1] + T(Wi(1, 2))*_xoi[2];
        S_xoi[2] = T(Wi(2, 0))*_xoi[0] + T(Wi(2, 1))*_xoi[1] + T(Wi(2, 2))*_xoi[2];
        T R_S_xoi[3];
        T q[4] = {Posei[6], Posei[3], Posei[4], Posei[5]};
        ceres::QuaternionRotatePoint(q, S_xoi, R_S_xoi);
        T mi[3] = {R_S_xoi[0]+V[0], R_S_xoi[1]+V[1], R_S_xoi[2]+V[2]};
        //todo
        
        T So_xoi[3];
        So_xoi[0] = T(Wo(0, 0))*_xoi[0] + T(Wo(0, 1))*_xoi[1] + T(Wo(0, 2))*_xoi[2];
        So_xoi[1] = T(Wo(1, 0))*_xoi[0] + T(Wo(1, 1))*_xoi[1] + T(Wo(1, 2))*_xoi[2];
        So_xoi[2] = T(Wo(2, 0))*_xoi[0] + T(Wo(2, 1))*_xoi[1] + T(Wo(2, 2))*_xoi[2];
        T Ro_So_xoi[3];
        T qo[4] = {T(_Qo.w()), T(_Qo.x()), T(_Qo.y()), T(_Qo.z())};
        ceres::QuaternionRotatePoint(qo, So_xoi, Ro_So_xoi);
        T vo[3] = {T(Vo[0])-Ro_So_xoi[0], T(Vo[1])-Ro_So_xoi[1], T(Vo[2])-Ro_So_xoi[2]};
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, vo, x_tr);
        vo[0] = x_tr[0];
        vo[1] = x_tr[1];
        vo[2] = x_tr[2];
        //todo

        residuals[0] = mi[0] -vo[0];
        residuals[1] = mi[1] - vo[1];
        residuals[2] = mi[2] - vo[2];
        //
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        return true;
    }

   static ceres::CostFunction* Create(Vector3d v, Quaterniond Qo, Vector3d w, Vector3d wo, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain_T_x, 3, 3, 7, 7, 3>(
	          	new KinematicVeloConstrain_T_x(v,Qo, w,wo, info)));
	}

    private:
    Quaterniond _Qo;
    Vector3d Vo;
    Matrix3d Wi;
    Matrix3d Wo;
    Matrix3d _info;
};


struct KinematicVeloConstrain_T_e2{
    KinematicVeloConstrain_T_e2(Vector3d x, Vector3d v, Vector3d c_w, Matrix3d info):Vo(v), _info(info){                
        Wo << 0, -c_w(2), c_w(1),
                    c_w(2), 0, -c_w(0),
                    -c_w(1), c_w(0), 0;
        xoi = Wo * x;
    }              
    template <typename T>
    bool operator()(const T* const V, const T* const Posei, const T* const _ET, T* residuals) const
    {
        //todo
        T vo[3] = {T(Vo[0]), T(Vo[1]), T(Vo[2])};
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, vo, x_tr);
        vo[0] = x_tr[0];
        vo[1] = x_tr[1];
        vo[2] = x_tr[2];
        //todo
        T x[3] = {vo[0]-V[0], vo[1]-V[1], vo[2]-V[2]};
        T mi[3];
        T q[4] = {Posei[6], -Posei[3], -Posei[4], -Posei[5]};
        ceres::QuaternionRotatePoint(q, x, mi);
      

        residuals[0] = mi[0] - T(xoi[0]);
        residuals[1] = mi[1] - T(xoi[1]);
        residuals[2] = mi[2] - T(xoi[2]);
        //
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain_T_e2, 3, 3, 7, 7>(
	          	new KinematicVeloConstrain_T_e2(x, v, w, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Vo;
    Matrix3d Wo;
    Matrix3d _info;
};

//手动求导
class KinematicPoseFactor : public ceres::SizedCostFunction<3, 7>
{
    public:
        KinematicPoseFactor(const Vector3d &x, const Vector3d &p, const Matrix3d &info);
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians)const;

    private:
        Vector3d xoi;
        Vector3d Po;
        Matrix3d _info;

};

class KinematicVeloFactor:public ceres::SizedCostFunction<3, 7, 3>
{
    public:
        KinematicVeloFactor(Vector3d x, Vector3d v, Vector3d c_w, Matrix3d info);
        virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians)const;
    
    private:
        Vector3d xoi;
        Vector3d Vo;
        Matrix3d Wo;
        Matrix3d _info;
};


//todo
//e2
struct KinematicPoseConstrain_e2{
    KinematicPoseConstrain_e2(Vector3d x, Vector3d p, Matrix3d info):
                                                        xoi(x), Po(p), _info(info){ }      
    template <typename T>
    bool operator()( const T* const Posei, T* residuals) const
    {
        //ROS_INFO("you are in 'openrator()'.");
        T _P[3] = {T(Po[0])-Posei[0], T(Po[1])-Posei[1], T(Po[2])-Posei[2]};
        //qi是子机姿态
        T qi_t[4] = {Posei[6], -Posei[3], -Posei[4], -Posei[5]};//wxyz
        T mi[3];
        ceres::QuaternionRotatePoint(qi_t, _P, mi);
        
        residuals[0] = mi[0] - T(xoi[0]);
        residuals[1] = mi[1] - T(xoi[1]);
        residuals[2] = mi[2] - T(xoi[2]);
        //加上信息矩阵
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
        //ROS_INFO("calculation done!!!!!");
        return true;
    }

    static ceres::CostFunction* Create(Vector3d x, Vector3d p, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicPoseConstrain_e2, 3, 7>(
	          	new KinematicPoseConstrain_e2(x, p, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Po;
    Matrix3d _info;
};




struct KinematicVeloConstrain_e2{
    KinematicVeloConstrain_e2(Vector3d x, Vector3d v, Vector3d c_w, Matrix3d info):xoi(x), Vo(v), _info(info){                
        Matrix3d S_wi;
        S_wi << 0, -c_w(2), c_w(1),
                    c_w(2), 0, -c_w(0),
                    -c_w(1), c_w(0), 0;
        xoi = S_wi*xoi;
    }              
    template <typename T>
    bool operator()(const T* const V, const T* const Posei, T* residuals) const
    {
        T _V[3] = {T(Vo[0])-V[0], T(Vo[1])-V[1], T(Vo[2])-V[2]};
        T R_V[3];
        T R_t[4] = {Posei[6], -Posei[3], -Posei[4], -Posei[5]};
        ceres::QuaternionRotatePoint(R_t, _V, R_V);
    
        residuals[0] = R_V[0] - T(xoi[0]);
        residuals[1] = R_V[1] - T(xoi[1]);
        residuals[2] = R_V[2] - T(xoi[2]);
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain_e2, 3, 3, 7>(
	          	new KinematicVeloConstrain_e2(x, v, w, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Vo;
    Matrix3d _info;
};
