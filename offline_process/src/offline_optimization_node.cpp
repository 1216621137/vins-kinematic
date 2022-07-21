#include <cstdio>
#include <vector>
#include <ros/ros.h>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <fstream>
#include "factor.h"
#include "bifactor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "initialize.h"
using namespace std;
using namespace Eigen;

string setting_file;
string save_path;
string save_path_j;
int e2;//残差的形式
int tr;//是否估计外参的标志位
Vector3d Xoi;
Vector3d Xoj;
double para_ET[7];
double x_oi[3] = {0, 0, 0};
double x_oj[3] = {0, 0, 0};
Matrix3d R;
Vector3d t;
Matrix3d rect_R;
Vector3d rect_t;
Matrix3d i_Simq = Matrix3d::Identity();
Vector3d i_Simt = Vector3d::Zero();
Matrix3d j_Simq = Matrix3d::Identity();
Vector3d j_Simt = Vector3d::Zero();
Matrix3d i_q_last = Matrix3d::Identity();
Vector3d  i_p_last = Vector3d::Zero();
Matrix3d j_q_last = Matrix3d::Identity();
Vector3d  j_p_last = Vector3d::Zero();
double weight;
double weight2;
double weight3;
double weight_k_p;
double weight_k_v;

int seq_n;
int  MAX_ITERATION_NUMBER;
int if_lossfunction;
int if_xoi;
int binary_path;
vector<pair<double, double>> match;//(m,n)  (center, child)

template<typename T>
T readParam(ros::NodeHandle &n, string name){
    T ans;
    //使用getParam函数读取launch里的参数
    if(n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded "<<name<<": "<<ans);
    }
    else{
        ROS_ERROR_STREAM("Failed to load "<<name);
        n.shutdown();
    }
    return ans;
}

struct Data
{
    Data(FILE *f)
    {
        if (fscanf(f, " %lf,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &t,
               &px, &py, &pz,
               &qw, &qx, &qy, &qz,
               &vx, &vy, &vz,
               &wx, &wy, &wz,
               &ax, &ay, &az) != EOF)
        {
            t /= 1e9;
        }
    }
    double t;
    float px, py, pz;
    float qw, qx, qy, qz;
    float vx, vy, vz;
    float wx, wy, wz;//原始数据中，这个值是零偏，我改了源码，输出了对应时刻的角速度
    float ax, ay, az;
};
//存数据的vector,child应该是只用关键帧的数据
vector<Data> center_path, child_path;
vector<Data>::iterator center;
vector<Data>::iterator child;
vector<Data>::iterator child_flag;
vector<Data>::iterator child_last;
vector<Data>::iterator center_flag;
vector<Data>::iterator center_last;
int n = 0;
int m =0;
int number = 10000000;//default value
int iterater_num = 0;
double cur_t = 0;
double last_t =0;
double Frequence = 10;//default value
int K =0;
int K_seq_i=0;
int K_seq_j=0;
int data_noise;
int Align ;
int processnumber;
int all_process;

int propagate;

int process(double para_Pose[][7], double para_SpeedBias[][9], 
                                    double rect_Pose[][7], double rect_SpeedBias[][9] )
{      
    K=0;
    ROS_INFO_STREAM("iterator time: "<<(++iterater_num));
    ROS_INFO_STREAM("start iterator index : "<<child - child_path.begin());
    ROS_INFO_STREAM("'n' index : "<<n);
    ROS_INFO_STREAM("child_last index : "<<child_last - child_path.begin());
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(0.1);
    
    if( (int)(child_path.end() - child_last+1)  < number){
        child_flag = child_path.end();
    }else{
        child_flag = child_last +1 + number;
    }
    ROS_INFO_STREAM("end index: "<<child_flag - child_path.begin());
    if(tr==1){
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_ET, 7, local_parameterization);
    }

    //先对齐
    for(; child != child_flag; center ++){    
        if(center == center_path.end() ||   child == child_path.end() ){
            if(center == center_path.end()) ROS_INFO("center is end!!!!!");
            if(child == child_path.end()) ROS_INFO("child is end!!!!!");
            break;
        }

        if(abs(center->t - child->t) < 0.0001 ){

            n = (int)(child - child_path.begin());
            para_Pose[n][0] = child->px; para_Pose[n][1] = child->py; para_Pose[n][2] = child->pz;
            para_Pose[n][3] = child->qx; para_Pose[n][4] = child->qy; para_Pose[n][5] = child->qz; para_Pose[n][6] = child->qw;
            para_SpeedBias[n][0] = child->vx; para_SpeedBias[n][1] =child->vy; para_SpeedBias[n][2] = child->vz;           
            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(para_Pose[n], 7, local_parameterization);
            problem.AddParameterBlock(para_SpeedBias[n], 3);
            //add sequential edge  
            Vector3d Pjs = {(child)->px, (child)->py, (child)->pz};
            Quaterniond Qjs = {(child)->qw, (child)->qx, (child)->qy, (child)->qz};
            for(int j = 1; j<seq_n; j++){//j太大就没意义了，他们的相对位置的值就不准了
                if(child - child_path.begin() >=j){
                    Vector3d Pis = {(child-j)->px, (child-j)->py, (child-j)->pz};
                    Quaterniond Qis = {(child-j)->qw, (child-j)->qx, (child-j)->qy, (child-j)->qz};
                    Vector3d pij = Qis.normalized().conjugate().toRotationMatrix()*( Pjs - Pis);
                    Quaterniond Qij = Qis.normalized().conjugate()*Qjs.normalized();
                    //ROS_INFO_STREAM("quaternion multiply: "<<Qij.x());
                    //Qij = Qis.toRotationMatrix().transpose()*Qjs.toRotationMatrix();
                    //ROS_INFO_STREAM("Matrix multiply: "<<Qij.x());
                    Vector3d i_v_ij = {child->vx - (child-j)->vx,
                                                child->vy - (child-j)->vy,
                                                child->vz - (child-j)->vz};
                    i_v_ij = Qis.normalized().conjugate().toRotationMatrix()*i_v_ij;
                    ceres::CostFunction *sequentfactor = sequnce_constrain::Create(pij, Qij, i_v_ij, weight, weight2,weight3);
                    problem.AddResidualBlock(sequentfactor,NULL, para_Pose[n-j], para_Pose[n], para_SpeedBias[n-j], para_SpeedBias[n]);
                }
            }
        
            if(n==0){
                ROS_INFO_STREAM("set constant :"<<0);
                problem.SetParameterBlockConstant(para_Pose[0]);
                problem.SetParameterBlockConstant(para_SpeedBias[0]); 
            }
            
            
            //降采样处理10hz
            cur_t = child->t;//计算的是对齐的数据的频率，就是可以添加运动学约束的频率
            if((1.0 / (cur_t - last_t) > Frequence) && last_t != 0 && Frequence !=0){
                child ++;
                continue;
                //ROS_INFO("skip!");
            }
            if(Frequence != 0 || child == child_flag-1 || center + 1 ==center_path.end()){
                K++;
                
                Matrix3d info_p =weight_k_p *  Matrix3d::Identity();
                Matrix3d info_v =weight_k_v *  Matrix3d::Identity();
                //info =  ER.inverse().transpose() *info;
                Vector3d Pos = {center->px, center->py, center->pz};
                Quaterniond Qos = {center->qw, center->qx, center->qy, center->qz};
                Qos = Qos.normalized();
                Vector3d Vos = {center->vx, center->vy, center->vz};
                Vector3d Wos = {center->wx, center->wy, center->wz};
                Vector3d angular_velocity_buf = {child->wx, child->wy, child->wz};//经过了零偏的修正
                
                if(e2 == 1){
                    if(tr == 1){
                        //pose
                        //ROS_INFO("tr!!!");
                        Vector3d Pose_o = (Pos - Qos*Xoi);
                        //ROS_INFO("kinematic constrain!");
                        ceres::CostFunction* pose_function = KinematicPoseConstrain_T_e2::Create(
                                                            Xoi, Pose_o, info_p);
                        if(if_lossfunction ==1){
                            problem.AddResidualBlock(pose_function,loss_function, para_Pose[n], para_ET);
                        }else{
                            problem.AddResidualBlock(pose_function,NULL, para_Pose[n], para_ET);
                        }
                        //参数大小不对。pose是一个7size的数组，但是要求要3size的数组
                        //velocity
                        
                        Matrix3d Swo;
                        Swo << 0, -Wos(2), Wos(1),
                                        Wos(2), 0, -Wos(0),
                                        -Wos(1), Wos(0), 0;
                        //Vector3d Vo =R* (Vos - Qos*Swo*Xoi);
                        Vector3d Vo = (Vos - Qos*Swo*Xoi);
                        
                        ceres::CostFunction* velo_function = KinematicVeloConstrain_T_e2::Create(
                                                            Xoi, Vo, angular_velocity_buf, info_v);
                        problem.AddResidualBlock(velo_function, loss_function, para_SpeedBias[n], para_Pose[n], para_ET);
                    
                    }else{
                        //pose
                        Vector3d Pose_o = Pos - Qos*Xoi;
                        if(data_noise != 1){
                                Pose_o = R*(Pos - Qos*Xoi)+t;}
                        //ROS_INFO("kinematic constrain!");
                        ceres::CostFunction* pose_function = KinematicPoseConstrain_e2::Create(
                                                            Xoi, Pose_o, info_p);
                        if(if_lossfunction ==1){
                            problem.AddResidualBlock(pose_function, loss_function, para_Pose[n]);
                        }else{
                            problem.AddResidualBlock(pose_function, NULL, para_Pose[n]);
                        }
                        //参数大小不对。pose是一个7size的数组，但是要求要3size的数组
                        //velocity
                        
                        Matrix3d Swo;
                        Swo << 0, -Wos(2), Wos(1),
                                        Wos(2), 0, -Wos(0),
                                        -Wos(1), Wos(0), 0;
                        
                        Vector3d Vo = (Vos - Qos*Swo*Xoi);
                        if(data_noise != 1){
                            Vo =R* (Vos - Qos*Swo*Xoi);
                        }
                        ceres::CostFunction* velo_function = KinematicVeloConstrain_e2::Create(
                                                            Xoi, Vo, angular_velocity_buf, info_v);
                        if(if_lossfunction ==1){
                            problem.AddResidualBlock(velo_function, loss_function, para_SpeedBias[n], para_Pose[n]);
                        }else{
                            problem.AddResidualBlock(velo_function, NULL, para_SpeedBias[n], para_Pose[n]);
                        }
                        
                    }
                }
                else{
                    if(if_xoi==1){
                        //pose
                            
                        Vector3d Pose_o = (Pos );
                        //ROS_INFO("kinematic constrain!");
                        ceres::CostFunction* pose_function = KinematicPoseConstrain_T_x::Create(
                                                            Pose_o, Qos, info_p);
                        if(if_lossfunction ==1){
                            problem.AddResidualBlock(pose_function,loss_function, para_Pose[n], para_ET, x_oi);
                        }else{
                            problem.AddResidualBlock(pose_function,NULL, para_Pose[n], para_ET, x_oi);
                        }
                        //参数大小不对。pose是一个7size的数组，但是要求要3size的数组
                        //velocity
                     
                        Vector3d Vo = (Vos );
                        
                        ceres::CostFunction* velo_function = KinematicVeloConstrain_T_x::Create(
                                                             Vo,Qos, angular_velocity_buf, Wos, info_v);
                        
                        if(if_lossfunction ==1){
                            problem.AddResidualBlock(velo_function, loss_function, para_SpeedBias[n], para_Pose[n], para_ET,x_oi);
                        }else{
                            problem.AddResidualBlock(velo_function, NULL, para_SpeedBias[n], para_Pose[n], para_ET,x_oi);
                        }
                       
                    
                    }else{
                        if(tr == 1){
                            //pose
                            
                            Vector3d Pose_o = (Pos - Qos*Xoi);
                            //ROS_INFO("kinematic constrain!");
                            ceres::CostFunction* pose_function = KinematicPoseConstrain_T::Create(
                                                                Xoi, Pose_o, info_p);
                            if(if_lossfunction ==1){
                                problem.AddResidualBlock(pose_function,loss_function, para_Pose[n], para_ET);
                            }else{
                                problem.AddResidualBlock(pose_function,NULL, para_Pose[n], para_ET);
                            }
                            //参数大小不对。pose是一个7size的数组，但是要求要3size的数组
                            //velocity
                            
                            Matrix3d Swo;
                            Swo << 0, -Wos(2), Wos(1),
                                            Wos(2), 0, -Wos(0),
                                            -Wos(1), Wos(0), 0;
                            //Vector3d Vo =R* (Vos - Qos*Swo*Xoi);
                            Vector3d Vo = (Vos - Qos*Swo*Xoi);
                            
                            ceres::CostFunction* velo_function = KinematicVeloConstrain_T::Create(
                                                                Xoi, Vo, angular_velocity_buf, info_v);
                            
                            if(if_lossfunction ==1){
                                problem.AddResidualBlock(velo_function, loss_function, para_SpeedBias[n], para_Pose[n], para_ET);
                            }else{
                                problem.AddResidualBlock(velo_function, NULL, para_SpeedBias[n], para_Pose[n], para_ET);
                            }
                    
                        }else{      
                            //pose
                            
                            Vector3d Pose_o = (Pos - Qos*Xoi);
                            if(data_noise != 1){
                                Pose_o = R*(Pos - Qos*Xoi)+t;}
                            //Vector3d euler_i = Utility::R2ypr(Qjs.toRotationMatrix());
                            //ROS_INFO("kinematic constrain!");
                            ceres::CostFunction* pose_function = KinematicPoseConstrain::Create(
                                                            Xoi, Pose_o, info_p);
                            if(if_lossfunction ==1){
                                problem.AddResidualBlock(pose_function, loss_function, para_Pose[n]);
                            }else{
                                problem.AddResidualBlock(pose_function, NULL, para_Pose[n]);
                            }
                            //参数大小不对。pose是一个7size的数组，但是要求要3size的数组
                            //velocity
                            
                            Matrix3d Swo;
                            Swo << 0, -Wos(2), Wos(1),
                                            Wos(2), 0, -Wos(0),
                                            -Wos(1), Wos(0), 0;
                            //
                            Vector3d Vo = (Vos - Qos*Swo*Xoi);
                            if(data_noise != 1){
                                Vo =R* (Vos - Qos*Swo*Xoi);
                            }
                            ceres::CostFunction* velo_function = KinematicVeloConstrain::Create(
                                                            Xoi, Vo, angular_velocity_buf, info_v);
                            if(if_lossfunction ==1){
                                problem.AddResidualBlock(velo_function, loss_function, para_SpeedBias[n], para_Pose[n]);
                            }else{
                                problem.AddResidualBlock(velo_function, NULL, para_SpeedBias[n], para_Pose[n]);
                            }
                            /***
                         //手动求导
                            Vector3d Pose_o = (Pos - Qos*Xoi);
                            if(data_noise != 1){
                                Pose_o = R*(Pos - Qos*Xoi)+t;}
                            //Vector3d Pose_o = (Pos - Qos*Xoi);
                            //Vector3d euler_i = Utility::R2ypr(Qjs.toRotationMatrix());
                            //ROS_INFO("kinematic constrain!");
                            KinematicPoseFactor *f_p = new KinematicPoseFactor(
                                                            Xoi, Pose_o, info);
                            problem.AddResidualBlock(f_p, NULL, para_Pose[n]);
                            //参数大小不对。pose是一个7size的数组，但是要求要3size的数组
                            //velocity
                            
                            Matrix3d Swo;
                            Swo << 0, -Wos(2), Wos(1),
                                            Wos(2), 0, -Wos(0),
                                            -Wos(1), Wos(0), 0;
                            //Vector3d Vo =R* (Vos - Qos*Swo*Xoi);
                            Vector3d Vo = (Vos - Qos*Swo*Xoi);
                            if(data_noise != 1){
                                Vo =R* (Vos - Qos*Swo*Xoi);
                            }
                            KinematicVeloFactor *f_v = new KinematicVeloFactor(
                                                            Xoi, Vo, angular_velocity_buf, info);
                            problem.AddResidualBlock(f_v, NULL, para_Pose[n],  para_SpeedBias[n]);
                            ***/
                        }
                    }
                   
                }
            }
            child ++;
            last_t = cur_t;
        }
    }
    //优化前的值
    i_q_last = Quaterniond{(child-1)->qw, (child-1)->qx, (child-1)->qy, (child-1)->qz}.normalized();
    i_p_last = Vector3d{(child-1)->px, (child-1)->py, (child-1)->pz};
    ROS_INFO_STREAM("residualblock number: "<<K);
    //开始优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = MAX_ITERATION_NUMBER;
    ceres::Solver::Summary summary;
    ROS_INFO("optimizating");
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    cout << summary.FullReport() << endl;
    //更新
    for(auto index = child_path.begin(); index<child; index++){
        int k = index - child_path.begin();
        for(int m = 0; m<7; m++)    rect_Pose[k][m] = para_Pose[k][m];
        for(int m = 0; m<3; m++)    rect_SpeedBias[k][m] = para_SpeedBias[k][m];
        if(if_xoi == 1)
            for(int m = 0; m<3; m++)    Xoi[m] = x_oi[m] ;
    }
    //计算优化前后的变化
    Matrix3d i_q_cur = Quaterniond{rect_Pose[n][6], rect_Pose[n][3], rect_Pose[n][4], rect_Pose[n][5]}.normalized().toRotationMatrix();
    Vector3d  i_p_cur = Vector3d{rect_Pose[n][0], rect_Pose[n][1], rect_Pose[n][2]};
    
    Quaterniond i_simq = Quaterniond(i_q_last.transpose()*i_q_cur); 
    i_Simq = i_simq.normalized();
    i_Simt = i_q_last.transpose()*(i_p_cur - i_p_last);
    /***
   Quaterniond simq = Quaterniond(q_cur*q_last.transpose()); 
    Simq = simq.normalized();
    Simt = (p_cur - Simq*p_last);
    ***/
    ROS_INFO_STREAM("child place: "<<int(child-child_path.begin()));
    ROS_INFO_STREAM("center place: "<<int(center-center_path.begin()));
    ROS_INFO_STREAM("Simq: "<<i_Simq);
    ROS_INFO_STREAM("Simt:"<<i_Simt);
    {
        ROS_INFO_STREAM("EX_t: "<<Vector3d(para_ET[0],para_ET[1],para_ET[2]));
        ROS_INFO_STREAM("delta value: "<< Vector3d(para_ET[0],para_ET[1],para_ET[2])-(t));
        ROS_INFO("EX_T:");
        Quaterniond Q_R;
        Q_R = R;
        Quaterniond ET_r = Quaterniond(para_ET[6],para_ET[3],para_ET[4],para_ET[5]).normalized();
        ROS_INFO_STREAM(" "<<ET_r.toRotationMatrix());
        Vector3d delta = 2 * (Q_R.normalized()* ET_r.conjugate()).vec();
        ROS_INFO("inital value:");
        ROS_INFO_STREAM(" "<<R);
        ROS_INFO("delta value:");
        ROS_INFO_STREAM(" "<<delta);
        ROS_INFO_STREAM("XOI_estimated: "<<Xoi);
    }
    if(center == center_path.end() ||   child_path.end() == child ){
            ROS_INFO("All data have optimized");
            return 0;
        }
    child_last = child-1;//-1是指处理了的位子
    if(all_process == 1){
        child = child_path.begin();
        center = center_path.begin();
    } 
    return 1;
}


int bin_process(double para_Pose[][7], double para_SpeedBias[][9], double rect_Pose[][7], double rect_SpeedBias[][9],
                                double para_Posej[][7], double para_SpeedBiasj[][9], double rect_Posej[][7], double rect_SpeedBiasj[][9] )
{   
    //cost  a:before optimization      b:after opyimization
    double a_seq_p_i = 0;
    double a_seq_p_j = 0;
    double b_seq_p_i = 0;
    double b_seq_p_j = 0;

    double a_seq_q_i = 0;
    double a_seq_q_j = 0;
    double b_seq_q_i = 0;
    double b_seq_q_j = 0;

    double a_seq_v_i = 0;
    double a_seq_v_j = 0;
    double b_seq_v_i = 0;
    double b_seq_v_j = 0;

    double a_seq_i = 0;
    double a_seq_j = 0;
    double b_seq_i = 0;
    double b_seq_j = 0;

    double a_kinematic_p = 0;
    double b_kinematic_p = 0;


    double a_kinematic_v = 0;
    double b_kinematic_v = 0;


    K=0;
    K_seq_i = 0;
    K_seq_j = 0;
    ROS_INFO_STREAM("iterator time: "<<(++iterater_num));
    ROS_INFO_STREAM("start iterator index : "<<child - child_path.begin());
    ROS_INFO_STREAM("'n's index : "<<n);
    ROS_INFO_STREAM("child_last index : "<<child_last - child_path.begin());
    ROS_INFO_STREAM("start iterator index : "<<center - center_path.begin());
    ROS_INFO_STREAM("'m's index : "<<m);
    ROS_INFO_STREAM("center_last index : "<<center_last - center_path.begin());
    ceres::Problem problem;
    ceres::LossFunction *loss_function = NULL;
    if(if_lossfunction == 1)    loss_function = new ceres::HuberLoss(0.1);
    
    if( (int)(child_path.end() - child_last+1)  < number){
        child_flag = child_path.end();
    }else{
        child_flag = child_last +1 + number;
    }
    ROS_INFO_STREAM("end index: "<<child_flag - child_path.begin());
    //赋初值，每次的初值是上一次优化后的结果，第一次是vio的输出

    /***
    for(int i = center-center_path.begin(); i<center_path.end()-center_path.begin(); i++)
    {
        
        para_Posej[i][0] = (center+i)->px; para_Posej[i][1] = (center+i)->py; para_Posej[i][2] = (center+i)->pz;
        para_Posej[i][3] = (center+i)->qx; para_Posej[i][4] = (center+i)->qy; para_Posej[i][5] = (center+i)->qz; para_Posej[i][6] = (center+i)->qw;
        para_SpeedBiasj[i][0] = (center+i)->vx; para_SpeedBiasj[i][1] =(center+i)->vy; para_SpeedBiasj[i][2] = (center+i)->vz;
        Eigen::Matrix3d Posej_r = Quaterniond(para_Posej[i][6], para_Posej[i][3], para_Posej[i][4], para_Posej[i][5]).toRotationMatrix();
        Eigen::Map<Eigen::Vector3d> Posej_t(para_Posej[i]);
        //坐标系装换，rect_R是两个vins坐标系的装换矩阵
        Posej_r = rect_R*Posej_r;
        Posej_t = rect_R*Posej_t + rect_t;
        //为什么t不覆盖？速度为啥不变？？？pvq三着的坐标为啥只变q

        Quaterniond Posej_q;
        Posej_q = Posej_r;
        para_Posej[i][3] =  Posej_q.x(); para_Posej[i][4] =  Posej_q.y(); para_Posej[i][5] =  Posej_q.z(); para_Posej[i][6] =  Posej_q.w();
        
       para_Posej[i][0] = rect_Posej[i][0]; para_Posej[i][1] =  rect_Posej[i][1]; para_Posej[i][2] =  rect_Posej[i][2];
        para_Posej[i][3] = rect_Posej[i][3]; para_Posej[i][4] =  rect_Posej[i][4]; para_Posej[i][5] =  rect_Posej[i][5]; para_Posej[i][6] =  rect_Posej[i][6];
        para_SpeedBiasj[i][0] = rect_SpeedBiasj[i][0]; para_SpeedBiasj[i][1] = rect_SpeedBiasj[i][1]; para_SpeedBiasj[i][2] = rect_SpeedBiasj[i][2];
        //添加参数块
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Posej[i], 7, local_parameterization);
        problem.AddParameterBlock(para_SpeedBiasj[i], 3);
    }

    ***/
    for(int i = child-child_path.begin(); i<child_flag-child_path.begin(); i++)
    {
        /***
        para_Pose[i][0] = (child+i)->px; para_Pose[i][1] = (child+i)->py; para_Pose[i][2] = (child+i)->pz;
        para_Pose[i][3] = (child+i)->qx; para_Pose[i][4] = (child+i)->qy; para_Pose[i][5] = (child+i)->qz; para_Pose[i][6] = (child+i)->qw;
        para_SpeedBias[i][0] = (child+i)->vx; para_SpeedBias[i][1] =(child+i)->vy; para_SpeedBias[i][2] = (child+i)->vz;
        ***/
       //用上次优化后的值作为初值
        para_Pose[i][0] = rect_Pose[i][0]; para_Pose[i][1] =  rect_Pose[i][1]; para_Pose[i][2] =  rect_Pose[i][2];
        para_Pose[i][3] = rect_Pose[i][3]; para_Pose[i][4] =  rect_Pose[i][4]; para_Pose[i][5] =  rect_Pose[i][5]; para_Pose[i][6] =  rect_Pose[i][6];
        para_SpeedBias[i][0] = rect_SpeedBias[i][0]; para_SpeedBias[i][1] = rect_SpeedBias[i][1]; para_SpeedBias[i][2] = rect_SpeedBias[i][2];
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], 7, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], 3);
    }
    if(all_process == 1 ){
        ROS_INFO("set constant : 0");
        //problem.SetParameterBlockConstant(para_Pose[0]);
        //problem.SetParameterBlockConstant(para_SpeedBias[0]); 
        //problem.SetParameterBlockConstant(para_Posej[0]);
        //problem.SetParameterBlockConstant(para_SpeedBiasj[0]); 
    }

    
    //先对齐
    for(; child != child_flag; center ++){    
        if(center == center_path.end() ||   child == child_path.end() ){
            if(center == center_path.end()) ROS_INFO("center is end!!!!!");
            if(child == child_path.end()) ROS_INFO("child is end!!!!!");
            break;
        }
        //给center的vector赋值,rect_P*的初值是vio的输出
        m = center - center_path.begin();
        para_Posej[m][0] = rect_Posej[m][0]; para_Posej[m][1] =  rect_Posej[m][1]; para_Posej[m][2] =  rect_Posej[m][2];
        para_Posej[m][3] = rect_Posej[m][3]; para_Posej[m][4] =  rect_Posej[m][4]; para_Posej[m][5] =  rect_Posej[m][5]; para_Posej[m][6] =  rect_Posej[m][6];
        para_SpeedBiasj[m][0] = rect_SpeedBiasj[m][0]; para_SpeedBiasj[m][1] = rect_SpeedBiasj[m][1]; para_SpeedBiasj[m][2] = rect_SpeedBiasj[m][2];
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Posej[m], 7, local_parameterization);
        problem.AddParameterBlock(para_SpeedBiasj[m], 3);
        //上次没有优化的变量才要乘以一个转换矩阵
        if(child>child_last || child_last == child_path.begin()){
            //上次没有优化到的变量，对应的rect_里面装的还是vio输出的值，是在vio坐标系下的值
            Eigen::Matrix3d Posej_r = Quaterniond(para_Posej[m][6], para_Posej[m][3], para_Posej[m][4], para_Posej[m][5]).toRotationMatrix();
            Eigen::Map<Eigen::Vector3d> Posej_t(para_Posej[m]);
            Eigen::Map<Eigen::Vector3d> Posej_v(para_SpeedBiasj[m]);
            //坐标系装换，rect_R是两个vins坐标系的装换矩阵
            Posej_r = rect_R*Posej_r;
            Posej_t = rect_R*Posej_t + rect_t;
            Posej_v = rect_R*Posej_v;
            Quaterniond Posej_q;
            Posej_q = Posej_r;
            para_Posej[m][0] = Posej_t.x(); para_Posej[m][1] = Posej_t.y(); para_Posej[m][2] = Posej_t.z();
            para_Posej[m][3] =  Posej_q.x(); para_Posej[m][4] =  Posej_q.y(); para_Posej[m][5] =  Posej_q.z(); para_Posej[m][6] =  Posej_q.w();
            para_SpeedBiasj[m][0] = Posej_v.x(); para_SpeedBiasj[m][1] = Posej_v.y(); para_SpeedBiasj[m][2] = Posej_v.z();
        }
       
       //sequential edge for center，是用vio的量进行计算的
        Vector3d j_Pjs = {(center)->px, (center)->py, (center)->pz};
        Quaterniond j_Qjs = {(center)->qw, (center)->qx, (center)->qy, (center)->qz};
        for(int j = 1; j<seq_n; j++){//j太大就没意义了，他们的相对位置的值就不准了
            if(center - center_path.begin() >=j ){
                K_seq_j++;
                Vector3d j_Pis = {(center-j)->px, (center-j)->py, (center-j)->pz};
                Quaterniond j_Qis = {(center-j)->qw, (center-j)->qx, (center-j)->qy, (center-j)->qz};
                Vector3d j_pij = j_Qis.conjugate()*( j_Pjs - j_Pis);
                Quaterniond j_Qij = j_Qis.conjugate()*j_Qjs;
                //ROS_INFO_STREAM("quaternion multiply: "<<Qij.x());
                //Qij = Qis.toRotationMatrix().transpose()*Qjs.toRotationMatrix();
                //ROS_INFO_STREAM("Matrix multiply: "<<Qij.x());
                Vector3d j_i_v_ij = {center->vx - (center-j)->vx,
                                            center->vy - (center-j)->vy,
                                            center->vz - (center-j)->vz};
                j_i_v_ij = j_Qis.normalized().conjugate().toRotationMatrix()*j_i_v_ij;
                ceres::CostFunction *sequentfactor = sequnce_constrain::Create(j_pij, j_Qij, j_i_v_ij, weight, weight2, weight3);
                problem.AddResidualBlock(sequentfactor,NULL, para_Posej[m-j], para_Posej[m], para_SpeedBiasj[m-j], para_SpeedBiasj[m]);
                //统计cost
                Vector3d value_pi = Vector3d(para_Posej[m-j][0], para_Posej[m-j][1], para_Posej[m-j][2]);
                Vector3d value_pj = Vector3d(para_Posej[m][0], para_Posej[m][1], para_Posej[m][2]);
                Quaterniond value_qi = {para_Posej[m-j][6], para_Posej[m-j][3], para_Posej[m-j][4], para_Posej[m-j][5]};
                Quaterniond value_qj = {para_Posej[m][6], para_Posej[m][3], para_Posej[m][4], para_Posej[m][5]};
                Vector3d value_vi = Vector3d(para_SpeedBiasj[m-j][0], para_SpeedBiasj[m-j][1], para_SpeedBiasj[m-j][2]);
                Vector3d value_vj = Vector3d(para_SpeedBiasj[m][0], para_SpeedBiasj[m][1], para_SpeedBiasj[m][2]);
                a_seq_p_j += p_cost(j_pij, value_pi, value_pj, value_qi, weight);
                //cout<<a_seq_p_j<<endl;
                a_seq_q_j += q_cost(j_Qij, value_qi, value_qj, weight2);
                //速度的残差的形式和p一样
                a_seq_v_j += p_cost(j_i_v_ij, value_vi, value_vj, value_qi, weight3);
            }
        }

        if(abs(center->t - child->t) < 0.0001 || center->t > child->t ){  
            //add sequential edge  从约束帧到上次优化的结束帧
            Vector3d Pjs = {(child)->px, (child)->py, (child)->pz};
            Quaterniond Qjs = {(child)->qw, (child)->qx, (child)->qy, (child)->qz};
            n = (int)(child - child_path.begin());
            //sequencial edge
            for(int j = 1; j<seq_n; j++){//j太大就没意义了，他们的相对位置的值就不准了
                if(child - child_path.begin()>=j ){
                    K_seq_i++;
                    Vector3d Pis = {(child-j)->px, (child-j)->py, (child-j)->pz};
                    Quaterniond Qis = {(child-j)->qw, (child-j)->qx, (child-j)->qy, (child-j)->qz};
                    Vector3d pij = Qis.normalized().conjugate().toRotationMatrix()*( Pjs - Pis);
                    Quaterniond Qij = Qis.normalized().conjugate()*Qjs.normalized();
                    Vector3d i_v_ij = {child->vx - (child-j)->vx,
                                                child->vy - (child-j)->vy,
                                                child->vz - (child-j)->vz};
                    i_v_ij = Qis.normalized().conjugate().toRotationMatrix()*i_v_ij;
                    ceres::CostFunction *sequentfactor = sequnce_constrain::Create(pij, Qij, i_v_ij, weight, weight2, weight3);
                    problem.AddResidualBlock(sequentfactor,NULL, para_Pose[n-j], para_Pose[n], para_SpeedBias[n-j], para_SpeedBias[n]);

                    Vector3d value_pi = Vector3d(para_Pose[n-j][0], para_Pose[n-j][1], para_Pose[n-j][2]);
                    Vector3d value_pj = Vector3d(para_Pose[n][0], para_Pose[n][1], para_Pose[n][2]);
                    Quaterniond value_qi = {para_Pose[n-j][6], para_Pose[n-j][3], para_Pose[n-j][4], para_Pose[n-j][5]};
                    Quaterniond value_qj = {para_Pose[n][6], para_Pose[n][3], para_Pose[n][4], para_Pose[n][5]};
                    Vector3d value_vi = Vector3d(para_SpeedBias[n-j][0], para_SpeedBias[n-j][1], para_SpeedBias[n-j][2]);
                    Vector3d value_vj = Vector3d(para_SpeedBias[n][0], para_SpeedBias[n][1], para_SpeedBias[n][2]);
                    a_seq_p_i += p_cost(pij, value_pi, value_pj, value_qi, weight);
                    a_seq_q_i += q_cost(Qij, value_qi, value_qj, weight2);
                    //速度的残差的形式和p一样
                    a_seq_v_i += p_cost(i_v_ij, value_vi, value_vj, value_qi, weight3);
                }
            }
            
            //降采样处理10hz
            cur_t = child->t;//计算的是对齐的数据的频率，就是可以添加运动学约束的频率
            if((1.0 / (cur_t - last_t) > Frequence) && last_t != 0 && Frequence !=0){
                child ++;
                continue;
                //ROS_INFO("skip!"); 
            }
            if(Frequence != 0 || child == child_flag-1 || center + 1 ==center_path.end()){
                K++;
                Matrix3d info_p =weight_k_p *  Matrix3d::Identity();
                Matrix3d info_v =weight_k_v *  Matrix3d::Identity();
                //info =  ER.inverse().transpose() *info;
                Vector3d j_w = {center->wx, center->wy, center->wz};
                Vector3d i_w = {child->wx, child->wy, child->wz};//经过了零偏的修正
                if(e2 == 1){
                    ROS_INFO("Hello thank you! thank you very much !");
                }
                else{
                    match.emplace_back(make_pair(m,n));
                    //pose
                    ceres::CostFunction* posefactor = Bin_KinematicPoseConstrain_T_x::Create(info_p);

                    problem.AddResidualBlock(posefactor, loss_function, para_Pose[n], x_oi, 
                                                                                                                        para_Posej[m], x_oj);
                    //velocity
                    ceres::CostFunction* velofactor = Bin_KinematicVeloConstrain_T_x::Create(i_w, j_w, info_v);
                    problem.AddResidualBlock(velofactor, loss_function, para_SpeedBias[n], para_Pose[n], x_oi, 
                                                                                                                        para_SpeedBiasj[m], para_Posej[m], x_oj);


                    //if(tr != 1) problem.SetParameterBlockConstant(para_ET);
                    if(if_xoi != 1){
                        problem.SetParameterBlockConstant(x_oi);
                        problem.SetParameterBlockConstant(x_oj);
                    
                    //计算cost
                    //p kinematic constrain
                    Vector3d value_pa = Vector3d(para_Pose[n][0], para_Pose[n][1], para_Pose[n][2]);
                    Quaterniond value_qa = {para_Pose[n][6], para_Pose[n][3], para_Pose[n][4], para_Pose[n][5]};
                    Vector3d value_pb = Vector3d(para_Posej[m][0], para_Posej[m][1], para_Posej[m][2]);
                    Quaterniond value_qb = {para_Posej[m][6], para_Posej[m][3], para_Posej[m][4], para_Posej[m][5]};
                    Vector3d po_a = value_pa + value_qa*Xoi;
                    Vector3d po_b = value_pb + value_qb*Xoj;
                    a_kinematic_p += kinematic_cost(po_a, po_b, weight_k_p);
                    //v cost
                    Vector3d value_va = Vector3d(para_SpeedBias[n][0], para_SpeedBias[n][1], para_SpeedBias[n][2]);
                    Vector3d value_vb = Vector3d(para_SpeedBiasj[m][0], para_SpeedBiasj[m][1], para_SpeedBiasj[m][2]);
                    Matrix3d S_iw, S_jw;
                    S_iw << 0, -i_w(2), i_w(1),
                                i_w(2), 0, -i_w(0),
                                -i_w(1), i_w(0), 0;
                    S_jw << 0, -j_w(2),j_w(1),
                                j_w(2), 0, -j_w(0),
                                -j_w(1), j_w(0), 0;
                    Vector3d vo_a = value_va + value_qa*S_iw*Xoi;
                    Vector3d vo_b = value_vb + value_qb*S_jw*Xoj;
                    a_kinematic_v += kinematic_cost(vo_a, vo_b, weight_k_v);
                    }   
                }
            }
            child ++;
            last_t = cur_t;
        }
    }
    //优化前的值
    i_q_last = Quaterniond{(child-1)->qw, (child-1)->qx, (child-1)->qy, (child-1)->qz}.normalized();
    i_p_last = Vector3d{(child-1)->px, (child-1)->py, (child-1)->pz};
    j_q_last = Quaterniond{(center-1)->qw, (center-1)->qx, (center-1)->qy, (center-1)->qz}.normalized();
    j_p_last = Vector3d{(center-1)->px, (center-1)->py, (center-1)->pz};
    ROS_INFO_STREAM("kinematic residualblock number: "<<K*2);
    ROS_INFO_STREAM("seq_i residualblock number: "<<K_seq_i);
    ROS_INFO_STREAM("seq_j residualblock number: "<<K_seq_j);
    //开始优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;//输出每一步的信息
    options.max_num_iterations = MAX_ITERATION_NUMBER;
    ceres::Solver::Summary summary;
    ROS_INFO("optimizating");
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
    //cout<<summary.initial_cost<<endl<<summary.final_cost<<endl;
    //更新
    auto child_point = child;
    for(child = child_path.begin(); child != child_point; child++){
        int k = child - child_path.begin();
        for(int mm = 0; mm<7; mm++)    rect_Pose[k][mm] = para_Pose[k][mm];
        for(int mm = 0; mm<3; mm++)    rect_SpeedBias[k][mm] = para_SpeedBias[k][mm];
        if(if_xoi == 1){
            for(int mm = 0; mm<3; mm++)    Xoi[mm] = x_oi[mm];
            for(int mm = 0; mm<3; mm++)    Xoj[mm] = x_oj[mm];
        }
        Vector3d Pjs = {(child)->px, (child)->py, (child)->pz};
        Quaterniond Qjs = {(child)->qw, (child)->qx, (child)->qy, (child)->qz};
        //sequencial cost after optimization
        for(int j = 1; j<seq_n; j++){//j太大就没意义了，他们的相对位置的值就不准了
            if(child - child_path.begin()>=j ){
                K_seq_i++;
                Vector3d Pis = {(child-j)->px, (child-j)->py, (child-j)->pz};
                Quaterniond Qis = {(child-j)->qw, (child-j)->qx, (child-j)->qy, (child-j)->qz};
                Vector3d pij = Qis.normalized().conjugate().toRotationMatrix()*( Pjs - Pis);
                Quaterniond Qij = Qis.normalized().conjugate()*Qjs.normalized();
                Vector3d i_v_ij = {child->vx - (child-j)->vx,
                                            child->vy - (child-j)->vy,
                                            child->vz - (child-j)->vz};
                i_v_ij = Qis.normalized().conjugate().toRotationMatrix()*i_v_ij;
                Vector3d value_pi = Vector3d(para_Pose[k-j][0], para_Pose[k-j][1], para_Pose[k-j][2]);
                Vector3d value_pj = Vector3d(para_Pose[k][0], para_Pose[k][1], para_Pose[k][2]);
                Quaterniond value_qi = {para_Pose[k-j][6], para_Pose[k-j][3], para_Pose[k-j][4], para_Pose[k-j][5]};
                Quaterniond value_qj = {para_Pose[k][6], para_Pose[k][3], para_Pose[k][4], para_Pose[k][5]};
                Vector3d value_vi = Vector3d(para_SpeedBias[k-j][0], para_SpeedBias[k-j][1], para_SpeedBias[k-j][2]);
                Vector3d value_vj = Vector3d(para_SpeedBias[k][0], para_SpeedBias[k][1], para_SpeedBias[k][2]);
                b_seq_p_i += p_cost(pij, value_pi, value_pj, value_qi, weight);
                b_seq_q_i += q_cost(Qij, value_qi, value_qj, weight2);
                //速度的残差的形式和p一样
                b_seq_v_i += p_cost(i_v_ij, value_vi, value_vj, value_qi, weight3);
            }
        }
    }
    auto center_point = center;
    for(center = center_path.begin(); center != center_point; center++){
        int k = center - center_path.begin();
        for(int mm = 0; mm<7; mm++)    rect_Posej[k][mm] = para_Posej[k][mm];
        for(int mm = 0; mm<3; mm++)    rect_SpeedBiasj[k][mm] = para_SpeedBiasj[k][mm];
        //sequnetial cost after optimizatin
        Vector3d j_Pjs = {(center)->px, (center)->py, (center)->pz};
        Quaterniond j_Qjs = {(center)->qw, (center)->qx, (center)->qy, (center)->qz};
        for(int j = 1; j<seq_n; j++){
            if(center - center_path.begin() >=j ){
                Vector3d j_Pis = {(center-j)->px, (center-j)->py, (center-j)->pz};
                Quaterniond j_Qis = {(center-j)->qw, (center-j)->qx, (center-j)->qy, (center-j)->qz};
                Vector3d j_pij = j_Qis.conjugate()*( j_Pjs - j_Pis);
                Quaterniond j_Qij = j_Qis.conjugate()*j_Qjs;
                //ROS_INFO_STREAM("quaternion multiply: "<<Qij.x());
                //Qij = Qis.toRotationMatrix().transpose()*Qjs.toRotationMatrix();
                //ROS_INFO_STREAM("Matrix multiply: "<<Qij.x());
                Vector3d j_i_v_ij = {center->vx - (center-j)->vx,
                                            center->vy - (center-j)->vy,
                                            center->vz - (center-j)->vz};
                j_i_v_ij = j_Qis.normalized().conjugate().toRotationMatrix()*j_i_v_ij;
                //统计cost
                Vector3d value_pi = Vector3d(para_Posej[k-j][0], para_Posej[k-j][1], para_Posej[k-j][2]);
                Vector3d value_pj = Vector3d(para_Posej[k][0], para_Posej[k][1], para_Posej[k][2]);
                Quaterniond value_qi = {para_Posej[k-j][6], para_Posej[k-j][3], para_Posej[k-j][4], para_Posej[k-j][5]};
                Quaterniond value_qj = {para_Posej[k][6], para_Posej[k][3], para_Posej[k][4], para_Posej[k][5]};
                Vector3d value_vi = Vector3d(para_SpeedBiasj[k-j][0], para_SpeedBiasj[k-j][1], para_SpeedBiasj[k-j][2]);
                Vector3d value_vj = Vector3d(para_SpeedBiasj[k][0], para_SpeedBiasj[k][1], para_SpeedBiasj[k][2]);
                b_seq_p_j += p_cost(j_pij, value_pi, value_pj, value_qi, weight);
                //cout<<a_seq_p_j<<endl;
                b_seq_q_j += q_cost(j_Qij, value_qi, value_qj, weight2);
                //速度的残差的形式和p一样
                b_seq_v_j += p_cost(j_i_v_ij, value_vi, value_vj, value_qi, weight3);
            }
        }
    }
    //calculate kinematic cost after optimization
    {
        for(auto index_pair = match.begin(); index_pair<match.end(); index_pair++){
            int n = index_pair->second;
            int m = index_pair->first;
            Vector3d j_w = {(center_path.begin()+m)->wx, (center_path.begin()+m)->wy, (center_path.begin()+m)->wz};
            Vector3d i_w = {(child_path.begin()+n)->wx, (child_path.begin()+n)->wy, (child_path.begin()+n)->wz};//经过了零偏的修正

            Vector3d value_pa = Vector3d(para_Pose[n][0], para_Pose[n][1], para_Pose[n][2]);
            Quaterniond value_qa = {para_Pose[n][6], para_Pose[n][3], para_Pose[n][4], para_Pose[n][5]};
            Vector3d value_pb = Vector3d(para_Posej[m][0], para_Posej[m][1], para_Posej[m][2]);
            Quaterniond value_qb = {para_Posej[m][6], para_Posej[m][3], para_Posej[m][4], para_Posej[m][5]};
            Vector3d po_a = value_pa + value_qa*Xoi;
            Vector3d po_b = value_pb + value_qb*Xoj;
            b_kinematic_p += kinematic_cost(po_a, po_b, weight_k_p);
            //v cost
            Vector3d value_va = Vector3d(para_SpeedBias[n][0], para_SpeedBias[n][1], para_SpeedBias[n][2]);
            Vector3d value_vb = Vector3d(para_SpeedBiasj[m][0], para_SpeedBiasj[m][1], para_SpeedBiasj[m][2]);
            Matrix3d S_iw, S_jw;
            S_iw << 0, -i_w(2), i_w(1),
                        i_w(2), 0, -i_w(0),
                        -i_w(1), i_w(0), 0;
            S_jw << 0, -j_w(2),j_w(1),
                        j_w(2), 0, -j_w(0),
                        -j_w(1), j_w(0), 0;
            Vector3d vo_a = value_va + value_qa*S_iw*Xoi;
            Vector3d vo_b = value_vb + value_qb*S_jw*Xoj;
            b_kinematic_v += kinematic_cost(vo_a, vo_b, weight_k_v);
        }
    }
    

    //输出cost的值   
    //ceres的cost的输出是指每一项costfunction的二笵数的0.5倍的和
    cout<<"------------------------------------------------------------------------------------------------------------"<<endl;
    cout << setiosflags(ios::left) << setw(10) << " " << resetiosflags(ios::left) // 用完之后清除
        << setiosflags(ios::right)<< setw(15) << "iteration_n" << setw(15) << "seq_p" << setw(15) << "seq_v"<< setw(15) << "seq_q"<< setw(15) << "kine_p"<< setw(15) << "kine_v"
        << setw(15) << "sum_seq_k"<< setw(15) << "sum_seq"<< setw(15) << "sum_kine"<< setw(15) << "sum"
        << resetiosflags(ios::right) << endl;
    cout<<"------------------------------------------------------------------------------------------------------------"<<endl;
    //before optimization
    cout<< setiosflags(ios::left) << setw(10) << "before_i" << resetiosflags(ios::left) // 用完之后清除
        << setiosflags(ios::right) << setw(15) <<iterater_num<< setw(15) << a_seq_p_i << setw(15) <<a_seq_v_i<< setw(15) << a_seq_q_i<< setw(15) << a_kinematic_p<< setw(15) << a_kinematic_v
        //sum_seq_k
        << setw(15) << (a_seq_p_i)+(a_seq_q_i)+(a_seq_v_i)
        //sum_seq
        << setw(15) << (a_seq_p_i+a_seq_p_j)+(a_seq_q_i+a_seq_q_j)+(a_seq_v_i+a_seq_v_j) 
        //sum_kine
        << setw(15) <<  a_kinematic_p + a_kinematic_v
        //sum
        << setw(15) << (a_seq_p_i+a_seq_p_j)+(a_seq_q_i+a_seq_q_j)+(a_seq_v_i+a_seq_v_j) + a_kinematic_p + a_kinematic_v
        << resetiosflags(ios::right) << endl;
    cout<< setiosflags(ios::left) << setw(10) << "before_j" << resetiosflags(ios::left) // 用完之后清除
        << setiosflags(ios::right) << setw(15) << " "<< setw(15) << a_seq_p_j << setw(15) <<a_seq_v_j<< setw(15) << a_seq_q_j
        //sum_seq_k
        << setw(15)<<" "<< setw(15 )<<" "<< setw(15)<< (a_seq_p_j)+(a_seq_q_j)+(a_seq_v_j)
        << resetiosflags(ios::right) << endl;

    //after optimization
    cout<< setiosflags(ios::left) << setw(10) << "after_i" << resetiosflags(ios::left) // 用完之后清除
        << setiosflags(ios::right) << setw(15) <<" "<< setw(15) << b_seq_p_i << setw(15) <<b_seq_v_i<< setw(15) << b_seq_q_i<< setw(15) << b_kinematic_p<< setw(15) << b_kinematic_v
        //sum_seq_k
        << setw(15) << (b_seq_p_i)+(b_seq_q_i)+(b_seq_v_i)
        //sum_seq
        << setw(15) << (b_seq_p_i+b_seq_p_j)+(b_seq_q_i+b_seq_q_j)+(b_seq_v_i+b_seq_v_j) 
        //sum_kine
        << setw(15) <<  b_kinematic_p + b_kinematic_v
        //sum
        << setw(15) << (b_seq_p_i+b_seq_p_j)+(b_seq_q_i+b_seq_q_j)+(b_seq_v_i+b_seq_v_j) + b_kinematic_p + b_kinematic_v
        << resetiosflags(ios::right) << endl;
    cout<< setiosflags(ios::left) << setw(10) << "after_j" << resetiosflags(ios::left) // 用完之后清除
        << setiosflags(ios::right) << setw(15) << " "<< setw(15) << b_seq_p_j << setw(15) <<b_seq_v_j<< setw(15) << b_seq_q_j
        //sum_seq_k
        << setw(15)<<" "<< setw(15 )<<" "<< setw(15)<< (b_seq_p_j)+(b_seq_q_j)+(b_seq_v_j)
        << resetiosflags(ios::right) << endl;
    cout<<"------------------------------------------------------------------------------------------------------------"<<endl;
    //reset match
    match.clear();
    //计算优化前后的变化
    Matrix3d i_q_cur = Quaterniond{rect_Pose[n][6], rect_Pose[n][3], rect_Pose[n][4], rect_Pose[n][5]}.normalized().toRotationMatrix();
    Vector3d i_p_cur = Vector3d{rect_Pose[n][0], rect_Pose[n][1], rect_Pose[n][2]};
    Matrix3d j_q_cur = Quaterniond{rect_Posej[m][6], rect_Posej[m][3], rect_Posej[m][4], rect_Posej[m][5]}.normalized().toRotationMatrix();
    Vector3d j_p_cur = Vector3d{rect_Posej[m][0], rect_Posej[m][1], rect_Posej[m][2]};
    Quaterniond i_simq = Quaterniond(i_q_last.transpose()*i_q_cur); 
    Quaterniond j_simq = Quaterniond(j_q_last.transpose()*j_q_cur); 
    i_Simq = i_simq.normalized();
    i_Simt = i_q_last.transpose()*(i_p_cur - i_p_last);
    j_Simq = j_simq.normalized();
    j_Simt = j_q_last.transpose()*(j_p_cur - j_p_last);
    //更新外参
    if(tr==1)
    {
        Eigen::Matrix4d gt_T_body = Eigen::Matrix4d::Identity(); 
        Eigen::Matrix4d vio_T_body = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d vio_T_gt = Eigen::Matrix4d::Identity();
        gt_T_body.block<3, 3>(0, 0) = j_q_last;
        gt_T_body.block<3, 1>(0, 3) = j_p_last;
        vio_T_body.block<3, 3>(0, 0) = j_q_cur;
        vio_T_body.block<3, 1>(0, 3) = j_p_cur;
        vio_T_gt = vio_T_body * gt_T_body.inverse();
        

        rect_R = vio_T_gt.block<3,3>(0, 0);
        rect_t = vio_T_gt.block<3, 1>(0,3);
        ROS_INFO_STREAM("EX_t: "<<rect_t);
        ROS_INFO_STREAM("delta value: "<<rect_t-(t));
        Quaterniond Q_R;
        Q_R = R;
        Quaterniond ET_r;
        ET_r = rect_R;
        ROS_INFO_STREAM("ET_r: "<<ET_r.toRotationMatrix());
        Vector3d delta = 2 * (Q_R.normalized()* ET_r.conjugate()).vec();
        ROS_INFO("inital value:");
        ROS_INFO_STREAM(" "<<R);
        ROS_INFO("delta value:");
        ROS_INFO_STREAM(" "<<delta);
    }
    
    ROS_INFO_STREAM("child place: "<<int(child-child_path.begin()));
    ROS_INFO_STREAM("center place: "<<int(center-center_path.begin()));

    ROS_INFO_STREAM("i_Simq: "<<i_Simq);
    ROS_INFO_STREAM("i_Simt:"<<i_Simt);
    ROS_INFO_STREAM("j_Simq: "<<j_Simq);
    ROS_INFO_STREAM("j_Simt:"<<j_Simt);
    {
        /***
        ROS_INFO_STREAM("EX_t: "<<Vector3d(para_ET[0],para_ET[1],para_ET[2]));
        ROS_INFO_STREAM("delta value: "<< Vector3d(para_ET[0],para_ET[1],para_ET[2])-(t));
        ROS_INFO("EX_T:");
        Quaterniond Q_R;
        Q_R = R;
        Quaterniond ET_r = Quaterniond(para_ET[6],para_ET[3],para_ET[4],para_ET[5]).normalized();
        ROS_INFO_STREAM(" "<<ET_r.toRotationMatrix());
        Vector3d delta = 2 * (Q_R.normalized()* ET_r.conjugate()).vec();
        ROS_INFO("inital value:");
        ROS_INFO_STREAM(" "<<R);
        ROS_INFO("delta value:");
        ROS_INFO_STREAM(" "<<delta);
        ***/
        ROS_INFO_STREAM("XOi_estimated: "<<Xoi);
        ROS_INFO_STREAM("XOj_estimated: "<<Xoj);
    }
    if(center == center_path.end() ||   child_path.end() == child ){
            ROS_INFO("All data have optimized");
            return 0;
        }
    child_last = child-1;//-1是指处理了的位子
    center_last = center - 1;
    if(all_process == 1){
        child = child_path.begin();
        center = center_path.begin();
  
    } 
    return 1;
}



void save_result(double rect_Pose[][7], double rect_SpeedBias[][9],
                                    double rect_Posej[][7], double rect_SpeedBiasj[][9]){
        //保存优化结果
    ofstream foutC(save_path.c_str(), ios::app);
    
    child = child_path.begin();
    center = center_path.begin();
    Eigen::Quaterniond Q(R);
    Matrix3d RR;
    RR = Q.inverse();
    Vector3d tt = -RR*t;
    for(; child != child_path.end(); child++){
        n = (int)(child - child_path.begin()); 
        Vector3d align_p = Vector3d(rect_Pose[n][0], rect_Pose[n][1], rect_Pose[n][2]);
        Vector3d align_v = Vector3d(rect_SpeedBias[n][0], rect_SpeedBias[n][1], rect_SpeedBias[n][2]);
        Quaterniond align_q = Quaterniond(rect_Pose[n][6], rect_Pose[n][3],  rect_Pose[n][4], rect_Pose[n][5]);
        if(Align == 1 && data_noise==0){
            
            align_p = RR*align_p + tt;
            align_v = RR*align_v;
            align_q = RR*align_q;
        }
        
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << child->t * 1e9 << ",";
        foutC.precision(5);
        foutC <<align_p[0] << "," << align_p[1] << "," <<align_p[2] << ","
        << align_q.w() << "," <<  align_q.x() << "," <<  align_q.y() << "," <<  align_q.z() << "," 
        << align_v[0] << "," << align_v[1]  << "," <<align_v[2]  << ","
        << 0 << "," << 0 << "," <<0<< ","
        << 0<< "," <<0 << "," <<0 << endl;
    }
    if(binary_path == 1){
        ofstream foutC_j(save_path_j.c_str(), ios::app);
        for(; center != center_path.end(); center++){
            n = (int)(center - center_path.begin()); 
            Vector3d align_p = Vector3d(rect_Posej[n][0], rect_Posej[n][1], rect_Posej[n][2]);
            Vector3d align_v = Vector3d(rect_SpeedBiasj[n][0], rect_SpeedBiasj[n][1], rect_SpeedBiasj[n][2]);
            Quaterniond align_q = Quaterniond(rect_Posej[n][6], rect_Posej[n][3],  rect_Posej[n][4], rect_Posej[n][5]);           
            foutC_j.setf(ios::fixed, ios::floatfield);
            foutC_j.precision(0);
            foutC_j << center->t * 1e9 << ",";
            foutC_j.precision(5);
            foutC_j <<align_p[0] << "," << align_p[1] << "," <<align_p[2] << ","
            << align_q.w() << "," <<  align_q.x() << "," <<  align_q.y() << "," <<  align_q.z() << "," 
            << align_v[0] << "," << align_v[1]  << "," <<align_v[2]  << ","
            << 0 << "," << 0 << "," <<0<< ","
            << 0<< "," <<0 << "," <<0 << endl;
        }
        foutC_j.close();
        ROS_INFO_STREAM("j results have been saved at "<<save_path_j.c_str());
    }
    foutC.close();
    ROS_INFO_STREAM("results have been saved at "<<save_path.c_str());
    return;
    
}
void Param_reader(){
    cv::FileStorage fsSettings(setting_file.c_str(), cv::FileStorage::READ);
    fsSettings["output_name"]>>save_path;
    fsSettings["output_name_j"]>>save_path_j;
    tr = fsSettings["tr_flag"];
    ROS_INFO_STREAM("tr: "<<tr);
    e2 = fsSettings["e2_flag"];
    ROS_INFO_STREAM("e2: "<<e2);
    Frequence = fsSettings["Fre"];
    ROS_INFO_STREAM("Fre: "<<Frequence);
    number = fsSettings["Number"];
    ROS_INFO_STREAM("iteration magnitude: "<<number);
    seq_n = fsSettings["seq_n"];
    ROS_INFO_STREAM("sequential edge: "<<seq_n);
    data_noise = fsSettings["data_noise"];
    ROS_INFO_STREAM("using data_noise.csv: "<<data_noise);
    weight = fsSettings["w_p"];
    ROS_INFO_STREAM("sequential p weight: "<<weight);
    weight2 = fsSettings["w_q"];
    ROS_INFO_STREAM("sequential q weight: "<<weight2);
    weight3 = fsSettings["w_v"];
    ROS_INFO_STREAM("sequential v weight: "<<weight3);
    weight_k_p = fsSettings["w_k_p"];
    ROS_INFO_STREAM("kinematic p weight: "<<weight_k_p);
    weight_k_v = fsSettings["w_k_v"];
    ROS_INFO_STREAM("kinematic v weight: "<<weight_k_v);
    Align = fsSettings["align"];
    ROS_INFO_STREAM("align: "<<Align);
    processnumber = fsSettings["processnumber"];
    ROS_INFO_STREAM("process number: "<<processnumber);
    all_process = fsSettings["all_process"];
    ROS_INFO_STREAM("process all?: "<<all_process);
    propagate = fsSettings["propagate"];
    ROS_INFO_STREAM("propagate: "<<propagate);
    MAX_ITERATION_NUMBER = fsSettings["max_iteration_number"];
    ROS_INFO_STREAM("max_iteration_number: "<<MAX_ITERATION_NUMBER);
    if_lossfunction = fsSettings["if_lossfunction"];
    ROS_INFO_STREAM("if_lossfunction: "<<if_lossfunction);
    if_xoi = fsSettings["if_xoi"];
    ROS_INFO_STREAM("if_xoi: "<<if_xoi);
    binary_path = fsSettings["binary_path"];
    ROS_INFO_STREAM("binary_path: "<<binary_path);
    //R_imu_gt
    cv::Mat cv_ER, cv_ET, cv_xoi, cv_xoj;
    fsSettings["extrinsicR"] >> cv_ER;
    fsSettings["extrinsicT"] >> cv_ET;
    fsSettings["Xoi"] >> cv_xoi;
    fsSettings["Xoj"] >> cv_xoj;
    cv::cv2eigen(cv_ER, R);
    cv::cv2eigen(cv_ET, t);
    cv::cv2eigen(cv_xoi, Xoi);
    ROS_INFO_STREAM("Xoi:"<<Xoi);
    cv::cv2eigen(cv_xoj, Xoj);
    ROS_INFO_STREAM("Xoj:"<<Xoj);
    //优化初值
    x_oi[0] = Xoi.x(); x_oi[1] = Xoi.y(); x_oi[2] = Xoi.z();
    x_oj[0] = Xoj.x(); x_oj[1] = Xoj.y(); x_oj[2] = Xoj.z();
    Eigen::Quaterniond EQ(R);
    R = EQ.normalized();
    rect_R = R;
    rect_t = t;
    if(tr == 1){
        //rect_R = Eigen::Matrix3d::Identity();//初始的旋转矩阵很重要
        //rect_t = Eigen::Vector3d::Zero();
    }
    ROS_INFO_STREAM("R_imu_gt:"<<R);
    ROS_INFO_STREAM("t:"<<t);
}

int main(int argc, char **argv){
    ros::init(argc, argv, "offline_process");
    ros::NodeHandle nh("~");
    //通过launch文件读取数据文件的名字
    string center_csv = readParam<string>(nh, "center_data_name");
    string vins_csv = readParam<string>(nh, "child_data_name");
    string Fre;
    //读取yaml文件
    setting_file = readParam<string>(nh, "setting");
    FILE *center_f = fopen(center_csv.c_str(), "r");
    
    if(center_f == NULL){
        ROS_WARN("can't load center path.");
        return 0;
    }
    FILE *child_f =fopen(vins_csv.c_str(), "r");
    if(child_f == NULL){
        ROS_WARN("can't load child path. ");
        return 0;
    }
    //读取文件内容
    while(!feof(center_f))
        center_path.emplace_back(center_f);
    
    fclose(center_f);
    while(!feof(child_f))
        child_path.emplace_back(child_f);
    center_path.pop_back();
    child_path.pop_back();
    ROS_INFO("Data loaded:   center = %d, child = %d", (int)center_path.size(), (int)child_path.size());
    //读取参数
    Param_reader();
    Quaterniond Q = Quaterniond(R);
    para_ET[0] = t.x(); para_ET[1] = t.y(); para_ET[2] = t.z();
    para_ET[3] = Q.x(); para_ET[4] = Q.y(); para_ET[5] = Q.z(); para_ET[6] = Q.w();
    if(if_xoi == 1||tr == 1){
        //rect_R = Matrix3d::Identity();
        //rect_t.setZero();
    }
    
    
    
    //优化用的double数组
    double para_Pose[(int)child_path.size()][7];//xyz,qx,qy,qz,qw
    double para_SpeedBias[(int)child_path.size()][9];
    double rect_Pose[(int)child_path.size()][7];//xyz,qx,qy,qz,qw
    double rect_SpeedBias[(int)child_path.size()][9];

    double para_Posej[(int)center_path.size()][7];//xyz,qx,qy,qz,qw
    double para_SpeedBiasj[(int)center_path.size()][9];
    double rect_Posej[(int)center_path.size()][7];//xyz,qx,qy,qz,qw
    double rect_SpeedBiasj[(int)center_path.size()][9];
    
    child = child_path.begin(); 
    center = center_path.begin();
    for(; child != child_path.end(); child++){
        n = (int)(child - child_path.begin()); 
        //ROS_INFO_STREAM("n: "<<n);
        para_Pose[n][0] = child->px; para_Pose[n][1] = child->py; para_Pose[n][2] = child->pz;
        para_Pose[n][3] = child->qx; para_Pose[n][4] = child->qy; para_Pose[n][5] = child->qz; para_Pose[n][6] = child->qw;
        para_SpeedBias[n][0] = child->vx; para_SpeedBias[n][1] = child->vy; para_SpeedBias[n][2] = child->vz;
        for(int i =3;i<9;i++ ) para_SpeedBias[n][i] = 0;
        //rect是装优化后的值，初值也是给了vio的输出
        rect_Pose[n][0] = child->px; rect_Pose[n][1] = child->py; rect_Pose[n][2] = child->pz;
        rect_Pose[n][3] = child->qx; rect_Pose[n][4] = child->qy; rect_Pose[n][5] = child->qz; rect_Pose[n][6] = child->qw;
        rect_SpeedBias[n][0] = child->vx; rect_SpeedBias[n][1] = child->vy; rect_SpeedBias[n][2] = child->vz;
        for(int i =3;i<9;i++ ) rect_SpeedBias[n][i] = 0;
       
    }


    for(; center != center_path.end(); center++){
        n = (int)(center - center_path.begin()); 
        //ROS_INFO_STREAM("n: "<<n);
        para_Posej[n][0] = center->px; para_Posej[n][1] = center->py; para_Posej[n][2] = center->pz;
        para_Posej[n][3] = center->qx; para_Posej[n][4] = center->qy; para_Posej[n][5] = center->qz; para_Posej[n][6] = center->qw;
        para_SpeedBiasj[n][0] = center->vx; para_SpeedBiasj[n][1] = center->vy; para_SpeedBiasj[n][2] = center->vz;
        for(int i =3;i<9;i++ ) para_SpeedBiasj[n][i] = 0;
        rect_Posej[n][0] = center->px; rect_Posej[n][1] = center->py; rect_Posej[n][2] = center->pz;
        rect_Posej[n][3] = center->qx; rect_Posej[n][4] = center->qy; rect_Posej[n][5] = center->qz; rect_Posej[n][6] = center->qw;
        rect_SpeedBiasj[n][0] = center->vx; rect_SpeedBiasj[n][1] = center->vy; rect_SpeedBiasj[n][2] = center->vz;
        for(int i =3;i<9;i++ ) rect_SpeedBiasj[n][i] = 0;
    }

    ROS_INFO("initialization over!");
    child = child_path.begin();
    n = 0;
    child_flag = child;
    child_last = child;

    center = center_path.begin();
    center_flag = center;
    center_last = center;
    ROS_INFO_STREAM("child position: "<<child - child_path.begin());
    Initialization();

    if(binary_path == 0){
        while(process(para_Pose, para_SpeedBias, rect_Pose, rect_SpeedBias) ==1){
            //每次优化后，把没优化的部分进行修正
            if(propagate ==1){
                for(int i = n+1; i < child_path.size(); i++){
                    Matrix3d  rec_r= Quaterniond{(child_path.begin()+i)->qw, (child_path.begin()+i)->qx, (child_path.begin()+i)->qy, (child_path.begin()+i)->qz}.normalized().toRotationMatrix();
                    Vector3d  rec_t = Vector3d{(child_path.begin()+i)->px, (child_path.begin()+i)->py, (child_path.begin()+i)->pz};
                    Vector3d rec_v = Vector3d{(child_path.begin()+i)->vx, (child_path.begin()+i)->vy, (child_path.begin()+i)->vz};
                    
                    rec_t = rec_r*i_Simt +rec_t;
                    rec_r = rec_r*i_Simq;
                    rec_v = rec_v;
                    Quaterniond rec_q(rec_r);
                    rec_q = rec_q.normalized();
                 /***
                    rec_t = Simq*rec_t + Simt;
                    rec_r = Simq*rec_r;
                    rec_v = rec_v;
                    Quaterniond rec_q(rec_r);
                    rec_q = rec_q.normalized();
                    ***/
                    rect_Pose[i][0] = rec_t.x(); rect_Pose[i][1] =rec_t.y(); rect_Pose[i][2] = rec_t.z();
                    rect_Pose[i][3] = rec_q.x(); rect_Pose[i][4] = rec_q.y(); rect_Pose[i][5] = rec_q.z(); rect_Pose[i][6] = rec_q.w();
                    rect_SpeedBias[i][0] = rec_v.x(); rect_SpeedBias[i][1] = rec_v.y(); rect_SpeedBias[i][2] = rec_v.z();
                }
            }
            processnumber --;
            if(processnumber>=0)
                break;
        };
    }else{
        while(bin_process(para_Pose, para_SpeedBias, rect_Pose, rect_SpeedBias, 
                                                para_Posej, para_SpeedBiasj, rect_Posej, rect_SpeedBiasj) ==1){
            //每次优化后，把没优化的部分进行修正
            if(propagate ==1){
                for(int i = n+1; i < child_path.size(); i++){
                    Matrix3d  rec_r= Quaterniond{(child_path.begin()+i)->qw, (child_path.begin()+i)->qx, (child_path.begin()+i)->qy, (child_path.begin()+i)->qz}.normalized().toRotationMatrix();
                    Vector3d  rec_t = Vector3d{(child_path.begin()+i)->px, (child_path.begin()+i)->py, (child_path.begin()+i)->pz};
                    Vector3d rec_v = Vector3d{(child_path.begin()+i)->vx, (child_path.begin()+i)->vy, (child_path.begin()+i)->vz};
                    
                    rec_t = rec_r*i_Simt +rec_t;
                    rec_r = rec_r*i_Simq;
                    rec_v = rec_v;
                    Quaterniond rec_q(rec_r);
                    rec_q = rec_q.normalized();
                    /***
                    rec_t = Simq*rec_t + Simt;
                    rec_r = Simq*rec_r;
                    rec_v = rec_v;
                    Quaterniond rec_q(rec_r);
                    rec_q = rec_q.normalized();
                    ***/
                    rect_Pose[i][0] = rec_t.x(); rect_Pose[i][1] =rec_t.y(); rect_Pose[i][2] = rec_t.z();
                    rect_Pose[i][3] = rec_q.x(); rect_Pose[i][4] = rec_q.y(); rect_Pose[i][5] = rec_q.z(); rect_Pose[i][6] = rec_q.w();
                    rect_SpeedBias[i][0] = rec_v.x(); rect_SpeedBias[i][1] = rec_v.y(); rect_SpeedBias[i][2] = rec_v.z();
                }
                for(int i = m+1; i < center_path.size(); i++){
                    Matrix3d  rec_r= Quaterniond{(center_path.begin()+i)->qw, (center_path.begin()+i)->qx, (center_path.begin()+i)->qy, (center_path.begin()+i)->qz}.normalized().toRotationMatrix();
                    Vector3d  rec_t = Vector3d{(center_path.begin()+i)->px, (center_path.begin()+i)->py, (center_path.begin()+i)->pz};
                    Vector3d rec_v = Vector3d{(center_path.begin()+i)->vx, (center_path.begin()+i)->vy, (center_path.begin()+i)->vz};
                    
                    rec_t = rec_r*i_Simt +rec_t;
                    rec_r = rec_r*i_Simq;
                    rec_v = rec_v;
                    Quaterniond rec_q(rec_r);
                    rec_q = rec_q.normalized();
                    /***
                    rec_t = Simq*rec_t + Simt;
                    rec_r = Simq*rec_r;
                    rec_v = rec_v;
                    Quaterniond rec_q(rec_r);
                    rec_q = rec_q.normalized();
                    ***/
                    rect_Posej[i][0] = rec_t.x(); rect_Posej[i][1] =rec_t.y(); rect_Posej[i][2] = rec_t.z();
                    rect_Posej[i][3] = rec_q.x(); rect_Posej[i][4] = rec_q.y(); rect_Posej[i][5] = rec_q.z(); rect_Posej[i][6] = rec_q.w();
                    rect_SpeedBiasj[i][0] = rec_v.x(); rect_SpeedBiasj[i][1] = rec_v.y(); rect_SpeedBiasj[i][2] = rec_v.z();
                }
            }
            processnumber --;
            if(processnumber>=0)
                break;
        };
    }
    save_result(rect_Pose, rect_SpeedBias, rect_Posej, rect_SpeedBiasj);
    return 0;
}