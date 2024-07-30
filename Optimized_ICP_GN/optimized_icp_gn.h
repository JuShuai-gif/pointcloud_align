#pragma once
#include <eigen3/Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>


class OptimizedICPGN
{
public:
    // eigen自动内存对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    OptimizedICPGN();
    bool SetTargetCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & target_cloud_ptr);
    bool Match(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud_ptr,
                const Eigen::Matrix4f& predict_pose,
                pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_source_cloud_ptr,
                Eigen::Matrix4f& result_pose);
    // 设置
    float GetFitnessScore(float max_range = std::numeric_limits<float>::max())const;

    void SetMaxIterations(unsigned int iter);

    void SetMaxCorrespondDistance(float max_correspond_distance);

    void SetTransformationEpsilon(float transformaiont_epsilon);

    bool HasConverged()const;

private:
    unsigned int max_iteraionts_{};
    float max_correspond_distance_{},transformation_epsilon_{};
    bool has_converge_ = false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_ptr_ = nullptr;
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_ptr_ = nullptr;
    Eigen::Matrix4f final_transformation_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_flann_ptr_ = nullptr;

};



