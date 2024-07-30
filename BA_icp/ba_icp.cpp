#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

	virtual void computeError()
	{
		const g2o::VertexSE3Expmap *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
		// measurement is p, point is p'
		_error = _measurement - pose->estimate().map(_point);
	}

	virtual void linearizeOplus()
	{
		g2o::VertexSE3Expmap *pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
		g2o::SE3Quat T(pose->estimate());
		Eigen::Vector3d xyz_trans = T.map(_point);
		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double z = xyz_trans[2];

		_jacobianOplusXi(0, 0) = 0;
		_jacobianOplusXi(0, 1) = -z;
		_jacobianOplusXi(0, 2) = y;
		_jacobianOplusXi(0, 3) = -1;
		_jacobianOplusXi(0, 4) = 0;
		_jacobianOplusXi(0, 5) = 0;

		_jacobianOplusXi(1, 0) = z;
		_jacobianOplusXi(1, 1) = 0;
		_jacobianOplusXi(1, 2) = -x;
		_jacobianOplusXi(1, 3) = 0;
		_jacobianOplusXi(1, 4) = -1;
		_jacobianOplusXi(1, 5) = 0;

		_jacobianOplusXi(2, 0) = -y;
		_jacobianOplusXi(2, 1) = x;
		_jacobianOplusXi(2, 2) = 0;
		_jacobianOplusXi(2, 3) = 0;
		_jacobianOplusXi(2, 4) = 0;
		_jacobianOplusXi(2, 5) = -1;
	}

	bool read(std::istream &in) { return true; }
	bool write(std::ostream &out) const { return true; }

protected:
	Eigen::Vector3d _point;
};

void bundleAdjustment(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts1, const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts2, Eigen::Matrix4d &T)
{
	// 初始化g2o
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block; // pose维度为 6, landmark 维度为 3
																  // Block::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); // 线性方程求解器
																  // Block *solver_ptr = new Block(linearSolver);												 // 矩阵块求解器
																  // g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
																  // g2o::SparseOptimizer optimizer;
																  // optimizer.setAlgorithm(solver);
	std::unique_ptr<Block::LinearSolverType>
		linearSolver(new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
	Block *solver_ptr = new Block(unique_ptr<Block::LinearSolverType>(linearSolver)); // 矩阵块求解器
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));



	g2o::SparseOptimizer optimizer;

	optimizer.setAlgorithm(solver);

	// vertex
	g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap(); // camera pose
	pose->setId(0);
	pose->setEstimate(g2o::SE3Quat(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0)));
	optimizer.addVertex(pose);

	// edges
	int index = 1;
	std::vector<EdgeProjectXYZRGBDPoseOnly *> edges;
	for (size_t i = 0; i < pts1->size(); i++)
	{
		EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2->points[i].x, pts2->points[i].y, pts2->points[i].z));
		edge->setId(index);
		edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap *>(pose));
		edge->setMeasurement(Eigen::Vector3d(pts1->points[i].x, pts1->points[i].y, pts1->points[i].z));
		edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);
		optimizer.addEdge(edge);
		index++;
		edges.push_back(edge);
	}

	optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(10);

	T = Eigen::Isometry3d(pose->estimate()).matrix();
}

int main(int argc, char **argv)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_copy(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPCDFile("bunny1.pcd", *source_cloud);
	pcl::io::loadPCDFile("bunny3.pcd", *target_cloud);
	pcl::copyPointCloud(*source_cloud, *icp_cloud);
	pcl::copyPointCloud(*target_cloud, *target_copy);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	kdtree->setInputCloud(target_cloud);

	std::vector<int> indexs(source_cloud->size());
	int iters = 0;
	double error = std::numeric_limits<double>::infinity();

	while (error > 0.0001 && iters < 100)
	{
		iters++;
		double last_error = error;
		double err = 0.0;

		Eigen::Matrix4d transformation_matrix;
		bundleAdjustment(target_copy, icp_cloud, transformation_matrix);
		pcl::transformPointCloud(*icp_cloud, *icp_cloud, transformation_matrix);

		for (int i = 0; i < icp_cloud->size(); ++i)
		{
			std::vector<int> index(1);
			std::vector<float> distance(1);
			kdtree->nearestKSearch(icp_cloud->points[i], 1, index, distance);
			err = err + sqrt(distance[0]);
			indexs[i] = index[0];
		}
		pcl::copyPointCloud(*target_cloud, indexs, *target_copy);

		error = err / source_cloud->size();
		std::cout << "iters:" << iters << "  " << "error:" << error << std::endl
				  << std::endl;
		if (fabs(last_error - error) < 1e-6)
			break;
	}
	pcl::io::savePCDFileBinary("icp_cloud.pcd", *icp_cloud);

	// 可视化
	pcl::visualization::PCLVisualizer viewer("registration Viewer");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(source_cloud, 0, 255, 0); // 原始点云绿色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(target_cloud, 255, 0, 0); // 目标点云红色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h(icp_cloud, 0, 0, 255);	// 匹配好的点云蓝色

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
