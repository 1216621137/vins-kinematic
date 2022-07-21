#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "../utility/utility.h"
#include "ros/ros.h"
using namespace Eigen;
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};

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
        /**
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        ***/
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain, 3, 9, 7>(
	          	new KinematicVeloConstrain(x, v, w, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Vo;
    Matrix3d Wo;
    Matrix3d _info;
};

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
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, mi, x_tr);
        mi[0] = x_tr[0]+_ET[0];
        mi[1] = x_tr[1]+_ET[1];
        mi[2] = x_tr[2]+_ET[2];
        residuals[0] = mi[0] - T(Po[0]);
        residuals[1] = mi[1] - T(Po[1]);
        residuals[2] = mi[2] - T(Po[2]);
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
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T x_tr[3];
        ceres::QuaternionRotatePoint(q_e, mi, x_tr);
        mi[0] = x_tr[0];
        mi[1] = x_tr[1];
        mi[2] = x_tr[2];
        //todo

        residuals[0] = mi[0] - T(Vo[0]);
        residuals[1] = mi[1] - T(Vo[1]);
        residuals[2] = mi[2] - T(Vo[2]);
        //
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain_T, 3, 9, 7, 7>(
	          	new KinematicVeloConstrain_T(x, v, w, info)));
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

class KinematicVeloFactor:public ceres::SizedCostFunction<3, 7, 9>
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
        Swi = S_wi;
    }              
    template <typename T>
    bool operator()(const T* const V, const T* const Posei, T* residuals) const
    {
        T _V[3] = {T(Vo[0])-V[0], T(Vo[1])-V[1], T(Vo[2])-V[2]};
        T R_V[3];
        T R_t[4] = {Posei[6], -Posei[3], -Posei[4], -Posei[5]};
        ceres::QuaternionRotatePoint(R_t, _V, R_V);
        T s[3];
        T Swi_t[4] ={T(Swi.w()), -T(Swi.x()), -T(Swi.y()), -T(Swi.z())};
        ceres::QuaternionRotatePoint(Swi_t, R_V, s);
        
        residuals[0] = s[0] - T(xoi[0]);
        residuals[1] = s[1] - T(xoi[1]);
        residuals[2] = s[2] - T(xoi[2]);
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        return true;
    }

   static ceres::CostFunction* Create(Vector3d x, Vector3d v, Vector3d w, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          KinematicVeloConstrain_e2, 3, 9, 7>(
	          	new KinematicVeloConstrain_e2(x, v, w, info)));
	}

    private:
    Vector3d xoi;
    Vector3d Vo;
    Quaterniond Swi;
    Matrix3d _info;
};

