#include <iostream>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h> 
#include <pcl/correspondence.h>
#include <pcl/visualization/pcl_visualizer.h>

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
	double error = std::numeric_limits<double>::infinity();
    // 旋转矩阵
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	// 位移
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
		double last_error = error;
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

		if (fabs(last_error - error) < 1e-6)
			break;

		//计算点云中心坐标
		Eigen::Vector4f source_centroid, target_centroid_mid;
		pcl::compute3DCentroid(*source_cloud_mid, source_centroid);
		pcl::compute3DCentroid(*target_cloud_mid, target_centroid_mid);

		//去中心化
		Eigen::MatrixXf souce_cloud_demean, target_cloud_demean;
		pcl::demeanPointCloud(*source_cloud_mid, source_centroid, souce_cloud_demean);
		pcl::demeanPointCloud(*target_cloud_mid, target_centroid_mid, target_cloud_demean);

		//计算W=q1*q2^T
		Eigen::MatrixXf W = (souce_cloud_demean * target_cloud_demean.transpose()).topLeftCorner(3, 3);

		//SVD分解得到新的旋转矩阵和平移矩阵
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::MatrixXf U = svd.matrixU();
		Eigen::MatrixXf V = svd.matrixV();

		if (U.determinant() * V.determinant() < 0)
		{
			for (int x = 0; x < 3; ++x)
				V(x, 2) *= -1;
		}

		R = V * U.transpose();
		T = target_centroid_mid.head(3) - R * source_centroid.head(3);
		H << R, T, 0, 0, 0, 1;
		H_final = H * H_final; //更新变换矩阵	
	}
	transformation_matrix << H_final;

	clock_t end = clock();
	std::cout << end - start << "ms" << std::endl;
	std::cout << transformation_matrix << std::endl;

	//配准结果
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_cloud, *icp_cloud, transformation_matrix);
	pcl::io::savePCDFileBinary("icp_cloud.pcd", *icp_cloud);

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
