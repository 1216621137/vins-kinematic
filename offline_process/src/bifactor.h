#pragma once
#include "factor.h"

struct Bin_KinematicPoseConstrain_T_x{
    Bin_KinematicPoseConstrain_T_x(Matrix3d info):_info(info){ }      
    template <typename T>
    bool operator()( const T* const Posei, const T* const xoi,const T* const Posej, const T* const xoj, T* residuals) const
    {
        //i_x_oi
        T _xoi[3] = {xoi[0], xoi[1], xoi[2]};
        T x_i[3];
        //qi是子机姿态
        T qi[4] = {Posei[6], Posei[3], Posei[4],Posei[5]};//wxyz
        ceres::QuaternionRotatePoint(qi, _xoi, x_i);
        T mi[3] = {x_i[0]+Posei[0], x_i[1]+Posei[1], x_i[2]+Posei[2]};
        //
        T _xoj[3] = {xoj[0], xoj[1], xoj[2]};
        T x_j[3];
        T qj[4] = {Posej[6], Posej[3], Posej[4],Posej[5]};//wxyz
        ceres::QuaternionRotatePoint(qj, _xoj, x_j);
        T mj[3] = {x_j[0]+Posej[0], x_j[1]+Posej[1], x_j[2]+Posej[2]};
        /***
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T w_mj[3];
        ceres::QuaternionRotatePoint(q_e, mj, w_mj);
        w_mj[0] = w_mj[0]+_ET[0];
        w_mj[1] = w_mj[1]+_ET[1];
        w_mj[2] = w_mj[2]+_ET[2];
        ***/
        residuals[0] = mi[0] - mj[0];
        residuals[1] = mi[1] - mj[1];
        residuals[2] = mi[2] - mj[2];
        //加上信息矩阵
        
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        
        //ROS_INFO("calculation done!!!!!");
        return true;
    }

    static ceres::CostFunction* Create(Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          Bin_KinematicPoseConstrain_T_x, 3, 7, 3, 7, 3>(
	          	new Bin_KinematicPoseConstrain_T_x( info)));
	}

    private:
    Matrix3d _info;
};



struct Bin_KinematicVeloConstrain_T_x{
    Bin_KinematicVeloConstrain_T_x(Vector3d c_wi, Vector3d c_wj, Matrix3d info): _info(info){                
        Wi << 0, -c_wi(2), c_wi(1),
                    c_wi(2), 0, -c_wi(0),
                    -c_wi(1), c_wi(0), 0;
        Wj << 0, -c_wj(2), c_wj(1),
                        c_wj(2), 0, -c_wj(0),
                        -c_wj(1), c_wj(0), 0;
    }              
    template <typename T>
    bool operator()(const T* const Vi, const T* const Posei, const T* const xoi, 
                                    const T* const Vj, const T* const Posej, const T* const xoj, T* residuals) const
    {
        T _xoi[3] = {xoi[0], xoi[1], xoi[2]};
        T S_xoi[3];
        S_xoi[0] = T(Wi(0, 0))*_xoi[0] + T(Wi(0, 1))*_xoi[1] + T(Wi(0, 2))*_xoi[2];
        S_xoi[1] = T(Wi(1, 0))*_xoi[0] + T(Wi(1, 1))*_xoi[1] + T(Wi(1, 2))*_xoi[2];
        S_xoi[2] = T(Wi(2, 0))*_xoi[0] + T(Wi(2, 1))*_xoi[1] + T(Wi(2, 2))*_xoi[2];
        T R_S_xoi[3];
        T qi[4] = {Posei[6], Posei[3], Posei[4], Posei[5]};
        ceres::QuaternionRotatePoint(qi, S_xoi, R_S_xoi);
        T mi[3] = {R_S_xoi[0]+Vi[0], R_S_xoi[1]+Vi[1], R_S_xoi[2]+Vi[2]};
   
        T _xoj[3] = {xoj[0], xoj[1], xoj[2]};
        T S_xoj[3];
        S_xoj[0] = T(Wj(0, 0))*_xoj[0] + T(Wj(0, 1))*_xoj[1] + T(Wj(0, 2))*_xoj[2];
        S_xoj[1] = T(Wj(1, 0))*_xoj[0] + T(Wj(1, 1))*_xoj[1] + T(Wj(1, 2))*_xoj[2];
        S_xoj[2] = T(Wj(2, 0))*_xoj[0] + T(Wj(2, 1))*_xoj[1] + T(Wj(2, 2))*_xoj[2];
        T R_S_xoj[3];
        T qj[4] = {Posej[6], Posej[3], Posej[4], Posej[5]};
        ceres::QuaternionRotatePoint(qj, S_xoj, R_S_xoj);
        T mj[3] = {R_S_xoj[0]+Vj[0], R_S_xoj[1]+Vj[1], R_S_xoj[2]+Vj[2]};
        /***
        T q_e[4] = {_ET[6], _ET[3], _ET[4], _ET[5]};//wxyz
        T w_mj[3];
        ceres::QuaternionRotatePoint(q_e, mj, w_mj);
        ***/
        residuals[0] = mi[0] - mj[0];
        residuals[1] = mi[1] - mj[1];
        residuals[2] = mi[2] - mj[2];
        //
        residuals[0] = T(_info(0, 0))*residuals[0] + T(_info(0, 1))*residuals[1] + T(_info(0, 2))*residuals[2];
        residuals[1] = T(_info(1, 0))*residuals[0] + T(_info(1, 1))*residuals[1] + T(_info(1, 2))*residuals[2];
        residuals[2] = T(_info(2, 0))*residuals[0] + T(_info(2, 1))*residuals[1] + T(_info(2, 2))*residuals[2];
        return true;
    }

   static ceres::CostFunction* Create(Vector3d c_wi, Vector3d c_wj, Matrix3d info) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          Bin_KinematicVeloConstrain_T_x, 3, 3, 7, 3, 3, 7, 3>(
	          	new Bin_KinematicVeloConstrain_T_x(c_wi, c_wj, info)));
	}

    private:
    Matrix3d Wi;
    Matrix3d Wj;
    Matrix3d _info;
};

