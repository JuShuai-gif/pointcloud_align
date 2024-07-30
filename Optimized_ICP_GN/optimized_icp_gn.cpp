#include "optimized_icp_gn.h"
#include "common.h"

OptimizedICPGN::OptimizedICPGN()
        :kdtree_flann_ptr_(new pcl::KdTreeFLANN<pcl::PointXYZ>)
{}

bool OptimizedICPGN::SetTargetCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & target_cloud_ptr)
{
   target_cloud_ptr_ = target_cloud_ptr;
       
   // 构建kdtree用于全局最近邻搜索
   kdtree_flann_ptr_->setInputCloud(target_cloud_ptr);std::cout << "成功设置目标点云！！！" << std::endl; 
   return true;
}

bool OptimizedICPGN::Match(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud_ptr,
                                const Eigen::Matrix4f& predict_pose,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_source_cloud_ptr,
                                Eigen::Matrix4f& result_pose)
{
        has_converge_ = false;
        source_cloud_ptr_ = source_cloud_ptr;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        Eigen::Matrix4f T = predict_pose;

        // 高斯牛顿方法解决ICP
        for (unsigned int i = 0; i < max_iteraionts_; ++i)
        {
                pcl::transformPointCloud(*source_cloud_ptr,*transformed_cloud,T);
                Eigen::Matrix<float,6,6> Hessian = Eigen::Matrix<float,6,6>::Zero();
                Eigen::Matrix<float,6,1> B = Eigen::Matrix<float,6,1>::Zero();

                for (unsigned int j = 0; j < transformed_cloud->size(); ++j)
                {
                        const pcl::PointXYZ& origin_point = source_cloud_ptr->points[j];
                        
                        // 删除距离为无穷的点
                        if (!pcl::isFinite(origin_point))
                        {
                                continue;
                        }
                        
                        const pcl::PointXYZ& transformed_point = transformed_cloud->at(j);
                        std::vector<float> resultant_distances;
                        std::vector<int> indices;
                        // 
                        kdtree_flann_ptr_->nearestKSearch(transformed_point,1,indices,resultant_distances);
                        
                        //
                        if (resultant_distances.front() > max_correspond_distance_)
                        {
                                continue;
                        }

                        Eigen::Vector3f nearest_point = Eigen::Vector3f(target_cloud_ptr_->at(indices.front()).x,
                                                                        target_cloud_ptr_->at(indices.front()).y,
                                                                        target_cloud_ptr_->at(indices.front()).z);
                                                                
                        Eigen::Vector3f point_eigen(transformed_point.x,transformed_point.y,transformed_point.z);
                        Eigen::Vector3f origin_point_eigen(origin_point.x,origin_point.y,origin_point.z);
                        Eigen::Vector3f error = point_eigen - nearest_point;

                        Eigen::Matrix<float,3,6> Jacobian = Eigen::Matrix<float,3,6>::Zero();

                        Jacobian.leftCols(3) = Eigen::Matrix3f::Identity();
                        Jacobian.rightCols(3) = -T.block<3,3>(0,0) * Hat(origin_point_eigen);

                        Hessian += Jacobian.transpose() * Jacobian;
                        B += -Jacobian.transpose() * error;
                }
                if (Hessian.determinant() == 0){
                        continue;
                }

                Eigen::Matrix<float,6,1> delte_x = Hessian.inverse() * B;

                T.block<3,1>(0,3) = T.block<3,1>(0,3) + delte_x.head(3);
                T.block<3,3>(0,0) *= SO3Exp(delte_x.tail(3)).matrix();

                if (delte_x.norm() < transformation_epsilon_){
                        has_converge_ = true;
                        break;
                }
        }
        final_transformation_ = T;
        result_pose = T;
        pcl::transformPointCloud(*source_cloud_ptr,*transformed_source_cloud_ptr,result_pose);
        return true;
}

float OptimizedICPGN::GetFitnessScore(float max_range)const
{
        float fitness_score = 0.0f;
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        // source_cloud_ptr_  原点云
        // transformed_cloud_ptr  变换后点云
        // final_transformation_   变换矩阵
        pcl::transformPointCloud(*source_cloud_ptr_,*transformed_cloud_ptr,final_transformation_);

        std::vector<int> nn_indices(1);
        std::vector<float> nn_dists(1);

        int nr = 0;

        for (unsigned int i = 0; i < transformed_cloud_ptr->size(); ++i)
        {
                kdtree_flann_ptr_->nearestKSearch(transformed_cloud_ptr->points[i],1,nn_indices,nn_dists);

                if (nn_dists.front() <= max_range)
                {
                        fitness_score += nn_dists.front();
                        nr++;
                }
        }

        if (nr > 0)
                return fitness_score / static_cast<float>(nr);
        else
                return (std::numeric_limits<float>::max());
}


bool OptimizedICPGN::HasConverged()const
{
        return has_converge_;
}

void OptimizedICPGN::SetMaxIterations(unsigned int iter)
{
        max_iteraionts_ = iter;
}

void OptimizedICPGN::SetMaxCorrespondDistance(float max_correspond_distance)
{
        max_correspond_distance_ = max_correspond_distance;
}

void OptimizedICPGN::SetTransformationEpsilon(float transformztion_epsilon)
{       
        transformation_epsilon_ = transformztion_epsilon;
}


