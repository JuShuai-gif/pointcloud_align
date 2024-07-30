#include <iostream>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h> 
#include <pcl/correspondence.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "sophus/se3.hpp"

int main(int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_mid(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_mid(new pcl::PointCloud<pcl::PointXYZ>);

	//加载点云
	pcl::io::loadPCDFile("../../sources/bunny1.pcd", *source_cloud);
	pcl::io::loadPCDFile("../../sources/bunny2.pcd", *target_cloud);

	//建立kd树
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	kdtree->setInputCloud(target_cloud);

	//初始化变换矩阵等参数
	int iters = 0;
	Sophus::SE3f pose_gn;
	double error = std::numeric_limits<double>::infinity();
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	Eigen::Vector3f T = Eigen::Vector3f::Zero();
	Eigen::Matrix4f H = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f H_final = H;
	Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
	pcl::copyPointCloud(*source_cloud, *source_cloud_mid);

	//开始迭代，直到满足条件
	clock_t start = clock();

	while (error > 0.0001 && iters < 100)
	{
		iters++;
		double err = 0.0;
		pcl::transformPointCloud(*source_cloud_mid, *source_cloud_mid, H);
		std::vector<int>indexs(source_cloud_mid->size());
		for (int i = 0; i < source_cloud_mid->size(); ++i)
		{
			std::vector<int>index(1);
			std::vector<float>distance(1);
			kdtree->nearestKSearch(source_cloud_mid->points[i], 1, index, distance);
			err = err + sqrt(distance[0]);
			indexs[i] = index[0];
		}
		pcl::copyPointCloud(*target_cloud, indexs, *target_cloud_mid);
		error = err / source_cloud->size();
		std::cout << "iters:" << iters << "  " << "error:" << error << std::endl;

		Eigen::Matrix<float, 6, 6> h = Eigen::Matrix<float, 6, 6>::Zero();
		Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
		for (int i = 0; i < target_cloud->size(); ++i)
		{
			Eigen::Vector3f p1_i = target_cloud_mid->points[i].getVector3fMap();
			Eigen::Vector3f p2_i = source_cloud_mid->points[i].getVector3fMap();
			Eigen::Vector3f p_n = pose_gn * p2_i;								// 经过pose转化后的点
			Eigen::Matrix<float, 3, 6> J = Eigen::Matrix<float, 3, 6>::Zero(); 	// 计算J
			J.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();					// 推导的jacobin中左边单位阵
			J.rightCols<3>() = -Sophus::SO3f::hat(p_n);							// 右边的李群
			J = -J;
			Eigen::Vector3f e = p1_i - p_n;										// 计算e
			h += J.transpose() * J;
			g += -J.transpose() * e;
		}

		Eigen::Matrix<float, 6, 1> dx = h.ldlt().solve(g);						// 求解dx
		pose_gn = Sophus::SE3f::exp(dx) * pose_gn;								// 进行更新，这里的dx是李代数
		H = pose_gn.matrix();
		H_final = H * H_final; //更新变换矩阵	
		//std::cout << H << std::endl;
	}
	transformation_matrix << H_final;

	clock_t end = clock();
	std::cout << end - start << "ms" << std::endl;
	std::cout << transformation_matrix << std::endl;

	//配准结果
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_cloud, *icp_cloud, transformation_matrix);
	pcl::io::savePCDFileBinary("icp.pcd", *icp_cloud);

	//可视化
	pcl::visualization::PCLVisualizer viewer("registration Viewer");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(source_cloud, 0, 255, 0); 	//原始点云绿色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(target_cloud, 255, 0, 0); 	//目标点云红色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h(icp_cloud, 0, 0, 255); 	//匹配好的点云蓝色

	viewer.setBackgroundColor(255, 255, 255);
	viewer.addPointCloud(source_cloud, src_h, "source cloud");
	viewer.addPointCloud(target_cloud, tgt_h, "target cloud");
	viewer.addPointCloud(icp_cloud, final_h, "result cloud");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
	}

	return 0;
}
