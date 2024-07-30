#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>

#include <Eigen/Dense>
#include "optimized_icp_gn.h"

int main()
{
    // 目标点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // 源点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // 优化得到点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_opti_transformed_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // SVD求解点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_svd_transformed_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // 目标点云
    pcl::io::loadPCDFile("../../sources/bunny1.pcd",*cloud_target_ptr);
    // 源点云
    pcl::io::loadPCDFile("../../sources/bunny2.pcd",*cloud_source_ptr);
    // 预测变换矩阵   最终变换矩阵
    Eigen::Matrix4f T_predict,T_final;
    T_predict.setIdentity();

    T_predict << 0.765,0.643,-0.027,-1.472,
                -0.644,0.765,-0.023,1.366,
                0.006,0.035,0.999,-0.125,
                0.0,0.0,0.0,1.0;

    std::cout << "Wait, matching..." << std::endl;

    OptimizedICPGN icp_opti;
    // 设置目标点云
    icp_opti.SetTargetCloud(cloud_target_ptr);
    // 设置变换矩阵误差
    icp_opti.SetTransformationEpsilon(1e-4);
    // 设置迭代次数
    icp_opti.SetMaxIterations(100);
    // 设置最大距离
    icp_opti.SetMaxCorrespondDistance(0.5);
    // 匹配
    icp_opti.Match(cloud_source_ptr,T_predict,cloud_source_opti_transformed_ptr,T_final);
    std::cout << "=================== Optimized ICP ===================" << std::endl;
    std::cout << "T final:\n" << T_final << std::endl;
    std::cout << "fitness score: " << icp_opti.GetFitnessScore() << std::endl;
    
    // 使用pcl自己自带的ICP 
    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp_svd;
    // 设置输入目标
    icp_svd.setInputTarget(cloud_target_ptr);
    // 设置输入源
    icp_svd.setInputSource(cloud_source_ptr);
    // 设置最大距离
    icp_svd.setMaxCorrespondenceDistance(0.5);
    // 设置迭代次数
    icp_svd.setMaximumIterations(100);
    // 设置误差范围
    icp_svd.setEuclideanFitnessEpsilon(1e-4);
    // 设置
    icp_svd.setTransformationEpsilon(1e-4);
    icp_svd.align(*cloud_source_svd_transformed_ptr,T_predict);
    std::cout << "\n=================== SVD ICP ==================" << std::endl;
    std::cout << "T final: \n" << icp_svd.getFinalTransformation() << std::endl;
    std::cout << "fitness score: " << icp_svd.getFitnessScore() << std::endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
    viewer->initCameraParameters();

    int v1(0);
    int v2(0);

    viewer->createViewPort(0.0,0.0,0.5,1.0,v1);
    viewer->setBackgroundColor(0,0,0,v1);
    viewer->addText("Optimized ICP",10,10,"Optimized icp",v1);
    
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_opti_color(
        cloud_source_opti_transformed_ptr,
        255,0,0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_source_opti_transformed_ptr,source_opti_color,"source opti cloud",
                                                v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_0(cloud_target_ptr,0,0,255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_target_ptr,target_color_0,"target cloud1",v2);

    viewer->createViewPort(0.5,0.0,1.0,1.0,v2);
    viewer->setBackgroundColor(0.0,0.0,0.0,v2);
    viewer->addText("SVD ICP",10,10,"svd icp",v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_1(cloud_target_ptr,0,0,255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_target_ptr,target_color_1,"target cloud2",v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_svd_color(cloud_source_svd_transformed_ptr,
                                                                                            0,255,0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_source_svd_transformed_ptr,source_svd_color,"source svd cloud",
                                            v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"source opti cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"source opti cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"source opti cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"source opti cloud");
    
    viewer->setCameraPosition(0,0,20,0,10,10,v1);
    viewer->setCameraPosition(0,0,20,0,10,10,v2);
    viewer->spin();

    return 0;
}









/*
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

int main() {
    // 创建一个输入点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 填充输入点云（这里仅为示例，实际应用中需要根据你的数据填充）
    input_cloud->width = 100;
    input_cloud->height = 1;
    input_cloud->points.resize(input_cloud->width * input_cloud->height);
    for (size_t i = 0; i < input_cloud->points.size(); ++i) {
        input_cloud->points[i].x = static_cast<float>(i);
        input_cloud->points[i].y = 0.0;
        input_cloud->points[i].z = 0.0;
    }

    // 定义变换矩阵
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    // 设置变换矩阵，这里进行一个平移操作
    transform_matrix(0, 3) = 1.0;  // x轴上平移1个单位

    // 创建一个输出点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 使用 transformPointCloud 函数进行坐标变换
    pcl::transformPointCloud(*input_cloud, *transformed_cloud, transform_matrix);

    // 输出变换后的点云
    std::cout << "Input Cloud Size: " << input_cloud->size() << std::endl;
    std::cout << "Transformed Cloud Size: " << transformed_cloud->size() << std::endl;

    return 0;
}
*/