#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Imu.h"
#include <opencv2/opencv.hpp>

#include "visensor7251/device.h"
#include "msckf/image_processor.h"
#include "msckf/msckf_vio.h"
//#include <msckf/CameraMeasurement.h>

using namespace cv;
using namespace std;

#define BASESEC 10000000
#define toRosTime(x) (ros::Time(BASESEC+x/100000000,(x-x/100000000*100000000)*10))

//supernode camera
sn::device *cam = NULL;

msckf::ImageProcessor *tracker;
msckf::MsckfVio *estimator;

/******************全局配置 start**********************/
int img_frame_rate = 20;
int imu_frame_rate = 200;
int auto_exposure = 1;
int exposure_time = 3;
int gain = 3;

//温漂补偿、常数噪声补偿、尺度偏差、轴间偏差
int temperature_compensation = 1;
int bias_correction = 1;
int scale_correction = 1;
int axis_alignment = 1;

//校正公式：a = T*K*(a_meas - a_bias - a_tempcompesation)
Mat tempdriftscale_acc,bias_acc,scale_acc,alignment_acc;
Mat tempdriftscale_gyr,bias_gyr,scale_gyr,alignment_gyr;

int cur_exposure_time = 0x1df0;
int max_exposure_time = 0x1f60;
int min_exposure_time = 0x1;
/******************全局配置 end**********************/


int readImuConfig(string config_file)
{
  cerr<<"open imu config file:"<<config_file<<endl;
  cerr << "---config settings---"<<endl;
  FileStorage fsettings(config_file, FileStorage::READ);
  if(!fsettings.isOpened()) {
    cerr<<"打开配置文件失败:"<<config_file<<endl;
    return -1;
  }

  img_frame_rate = (int)fsettings["img_frame_rate"];
  cerr << " img_frame_rate:"<<img_frame_rate <<endl;
  imu_frame_rate = (int)fsettings["imu_frame_rate"];
  cerr << " imu_frame_rate:"<<imu_frame_rate <<endl;
  auto_exposure = (int)fsettings["auto_exposure"];
  cerr << " auto_exposure:"<<auto_exposure <<endl;
  exposure_time = (int)fsettings["exposure_time"];
  cerr << " exposure_time:"<<exposure_time <<endl;
  gain = (int)fsettings["gain"];
  cerr << " gain:"<<gain <<endl;

  temperature_compensation = (int)fsettings["temperature_compensation"];
  cerr << " temperature_compensation:"<< temperature_compensation <<endl;
  if(temperature_compensation) {
    fsettings["tempdriftscale_acc"] >> tempdriftscale_acc;
    fsettings["tempdriftscale_gyr"] >> tempdriftscale_gyr;
  } else {
    tempdriftscale_acc = Mat::zeros(3,1,CV_64FC1);
    tempdriftscale_gyr = Mat::zeros(3,1,CV_64FC1);
  }

  bias_correction = (int)fsettings["bias_correction"];
  cerr<< " bias_correction:" << bias_correction <<endl;
  if(bias_correction) {
    fsettings["bias_acc"] >> bias_acc;
    fsettings["bias_gyr"] >> bias_gyr;
  } else {
    bias_acc = Mat::zeros(3,1,CV_64FC1);
    bias_gyr = Mat::zeros(3,1,CV_64FC1);
  }

  scale_correction = (int)fsettings["scale_correction"];
  cerr<< " scale_correction:" << scale_correction <<endl;
  if(scale_correction) {
    fsettings["Ka"] >> scale_acc;
    fsettings["Kg"] >> scale_gyr;    
  } else {
    // imu数据比例系数
    //double accS = 0.000244141 * 9.8 = 0.002392582; // m/s^2,量程+-8g
    //double gyrS = 0.06103515625*3.141592653/180.0 = 0.001065264; // rad/s
    scale_acc = Mat::eye(3,3,CV_64FC1) * 0.000244141 * 9.8;
    scale_gyr = Mat::eye(3,3,CV_64FC1) * 0.06103515625 * 3.141592653 / 180.0;
  }

  axis_alignment = (int)fsettings["axis_alignment"];
  cerr<< " axis_alignment:" << axis_alignment <<endl;
  if(axis_alignment) {
    fsettings["Ta"] >> alignment_acc;
    fsettings["Tg"] >> alignment_gyr;
  } else {
    alignment_acc = Mat::eye(3,3,CV_64FC1);
    alignment_gyr = Mat::eye(3,3,CV_64FC1);
  }

  cerr <<" tempdriftscale_acc:\n    "<<tempdriftscale_acc.t()<<endl;
  cerr <<" tempdriftscale_gyr:\n    "<<tempdriftscale_gyr.t()<<endl;
  cerr <<" bias_acc:\n    "<<bias_acc.t()<<endl;
  cerr <<" bias_gyr:\n    "<<bias_gyr.t()<<endl;
  cerr <<" scale_acc:\n"<<scale_acc<<endl;
  cerr <<" scale_gyr:\n"<<scale_gyr<<endl;
  cerr <<" alignment_acc:\n"<<alignment_acc<<endl;
  cerr <<" alignment_gyr:\n"<<alignment_gyr<<endl;
  cerr << "-----------------\n"<<endl;

  return 0;
}

//计算温漂
int get_temperature_drift(float temperature, float scale)
{
  //以50°为基准进行温度补偿
  float drift = (temperature - 50.0f) * scale;
  return int(drift);
}

//温漂
Mat get_temperature_drift(float temperature, Mat scale)
{
  //以50°为基准进行温度补偿
  Mat drift = (temperature - 50.0f) * scale;
  return drift;
}

void imu_process()
{
  sn::imuData imudata;
  
  while(1) {
    if(cam->readIMU(&imudata)) {
      //ROS_DEBUG("get imu...");

      //imu data
      float temperature = ((float)imudata.temperature) / 333.87f + 21.0f;

      //a = TK(a_meas - a_offset -a_tempcompesation)
      double acc_[3] = {(double)imudata.accel_x,(double)imudata.accel_y,(double)imudata.accel_z};
      double gyr_[3] = {(double)imudata.gyr_x,(double)imudata.gyr_y,(double)imudata.gyr_z};
      Mat meas_acc(3,1,CV_64FC1,acc_);
      Mat meas_gyr(3,1,CV_64FC1,gyr_);
      Mat temp_drift_acc = get_temperature_drift(temperature, tempdriftscale_acc);
      Mat temp_drift_gyr = get_temperature_drift(temperature, tempdriftscale_gyr);

      Mat acc = alignment_acc * scale_acc * (meas_acc - bias_acc - temp_drift_acc);
      Mat gyr = alignment_gyr * scale_gyr * (meas_gyr - bias_gyr - temp_drift_gyr);

      sensor_msgs::Imu imu_msg;
      ros::Time timestamp = toRosTime(imudata.imu_timestamp);
      imu_msg.header.frame_id = "imu";
      imu_msg.header.stamp = timestamp;
      imu_msg.angular_velocity.x = gyr.at<double>(0);
      imu_msg.angular_velocity.y = gyr.at<double>(1);
      imu_msg.angular_velocity.z = gyr.at<double>(2);
      imu_msg.linear_acceleration.x = acc.at<double>(0);
      imu_msg.linear_acceleration.y = acc.at<double>(1);
      imu_msg.linear_acceleration.z = acc.at<double>(2);

      //msckf imu process
      sensor_msgs::ImuPtr imu_msg_ptr(new sensor_msgs::Imu(imu_msg));
      tracker->imuCallback(imu_msg_ptr);
      estimator->imuCallback(imu_msg_ptr);
      
#if 0
      //predict
      predict(imu_msg);
      pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
#endif
    }
  }
}

void image_process()
{
  sn::resolution rn = cam->getImageSize();
  int width = rn.width;
  int height = rn.height;
  cv::Mat left8u = Mat(Size(width, height), CV_8UC1);
  cv::Mat right8u = Mat(Size(width, height), CV_8UC1);
  uint64 imgTimeStamp = 0;

  ROS_INFO("image loop start...");
  int nFrame = 0;
  while(1) {
    if (cam->QueryFrame(left8u, right8u, imgTimeStamp)) {
      if(nFrame++ < 5) //前几张图扔掉，时间戳有问题
	continue;

      ros::Time timestamp = toRosTime(imgTimeStamp);
      //ROS_DEBUG("get frame...");

      cv_bridge::CvImage cvi0;
      cvi0.header.stamp = timestamp;
      cvi0.header.frame_id = "image";
      cvi0.encoding = "mono8";
      cvi0.image = left8u;
      
      cv_bridge::CvImage cvi1;
      cvi1.header.stamp = timestamp;
      cvi1.header.frame_id = "image";
      cvi1.encoding = "mono8";
      cvi1.image = right8u;
      
      sensor_msgs::Image msg0,msg1;
      cvi0.toImageMsg(msg0);
      cvi1.toImageMsg(msg1);

      sensor_msgs::ImageConstPtr img_msg0( new sensor_msgs::Image(msg0));
      sensor_msgs::ImageConstPtr img_msg1( new sensor_msgs::Image(msg1));

      msckf::CameraMeasurementPtr feature_msg_ptr;
      feature_msg_ptr = tracker->stereoCallback(img_msg0,img_msg1);

      estimator->featureCallback(feature_msg_ptr);
    }
  }
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "msckf");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
  //ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  

  //read imu config file
  //string imu_config = "/home/sst/catkin_ws2/src/msckf/config/imu_50t3.yaml";
  //n.param<string>("imu_config",imu_config);
  //cerr<<imu_config.c_str()<<endl;
  //return -1;
  if(NULL == argv[1]) {
    fprintf(stderr,"parameter invalid,imu config needed.\n");
    return -1;
  }
  
  if(readImuConfig(string(argv[1]))){
    fprintf(stderr,"read imu config files failed.\n");
    return -1;
  }

  tracker = new msckf::ImageProcessor(n);
  estimator = new msckf::MsckfVio(n);

  //initialize
  tracker->initialize();
  estimator->initialize();
  
  // 1.open device
  cam = new sn::device();
  if (!cam->init()) {
    return 0;
  }

  // 2.device settings,帧率、自动曝光配置
  cam->config(img_frame_rate,imu_frame_rate,max_exposure_time,min_exposure_time,cur_exposure_time);

  std::thread thread_imu{imu_process};
  std::thread thread_image{image_process};

  // 3.capture image,必须先创建线程，再开启相机
  cam->start();

  // 
  ros::spin();

  //
  delete tracker;
  delete estimator;
  cam->close();
  delete cam;

  cerr<<"close camera."<<endl;

  //等待100ms
  usleep(1000*100);
  cerr<<"exit.."<<endl;

  return 0;
}
