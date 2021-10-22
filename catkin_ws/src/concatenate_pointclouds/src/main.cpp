
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_ros/transforms.h>

typedef sensor_msgs::PointCloud2 pc;
typedef const sensor_msgs::PointCloud2 ppc;

ros::Publisher pcl_pub;
tf2_ros::Buffer* tfBuffer;

void pointcloud_callback(const boost::shared_ptr<pc const>& frontLeft, const boost::shared_ptr<pc const>&frontRight, const boost::shared_ptr<pc const>&right, const boost::shared_ptr<pc const>&left, const boost::shared_ptr<pc const>& back)
{
//     const geometry_msgs::TransformStamped leftTrans = tfBuffer.lookupTransform(left->header.frame_id, "gpe",
//                                  left->header.stamp);

//     auto rightTrans = tfBuffer.lookupTransform(right->header.frame_id, "gpe",
//                                  right->header.stamp);

//     auto backTrans = tfBuffer.lookupTransform(back->header.frame_id, "gpe",
//                                  back->header.stamp);
    
//     auto frontleftTrans = tfBuffer.lookupTransform(frontLeft->header.frame_id, "gpe",
//                                  frontLeft->header.stamp);

//     auto frontrightTrans = tfBuffer.lookupTransform(frontRight->header.frame_id, "gpe",
//                                  frontRight->header.stamp);

    pc leftTransformed;
    pc rightTransformed;
    pc backTransformed;
    pc frontLeftTransformed;
    pc frontRightTransformed;
    std::string target_frame = "base_link";
    pcl_ros::transformPointCloud(target_frame, *left, leftTransformed, *tfBuffer);
    pcl_ros::transformPointCloud(target_frame, *right, rightTransformed, *tfBuffer);
    pcl_ros::transformPointCloud(target_frame, *back, backTransformed, *tfBuffer);
    pcl_ros::transformPointCloud(target_frame, *frontLeft, frontLeftTransformed, *tfBuffer);
    pcl_ros::transformPointCloud(target_frame, *frontRight, frontRightTransformed, *tfBuffer);

   
    sensor_msgs::PointCloud2 out;
    if (pcl::concatenatePointCloud(frontLeftTransformed, frontRightTransformed, out))
        {
	    if(pcl::concatenatePointCloud(out, rightTransformed, out))
             {
                 if(pcl::concatenatePointCloud(out, leftTransformed, out))
                 {
                    if(pcl::concatenatePointCloud(out, backTransformed, out))
                    {
                        pcl_pub.publish(out);
                    }
                 }
             }
	}
	else{
        ROS_ERROR_STREAM("cannot concat fleft, fright \n\n");
}
}


int main( int argc, char** argv)
{
    ros::init(argc, argv, "concatenate_pointclouds");
    ros::NodeHandle nh;
    pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/concatenated_pointcloud", 1);
    // ros::Subscriber<pc> testSub(nh, "/pointcloud/frontLeft")
    tfBuffer = new tf2_ros::Buffer();
    tf2_ros::TransformListener tfListener(*tfBuffer);

    message_filters::Subscriber<sensor_msgs::PointCloud2> frontLeftSub(nh, "/pointcloud/frontleft", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> frontRightSub(nh, "/pointcloud/frontright", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> rightSub(nh, "/pointcloud/right", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> leftSub(nh, "/pointcloud/left", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> backSub(nh, "/pointcloud/back", 1);

    typedef message_filters::sync_policies::ApproximateTime<pc, pc, pc, pc, pc> MySyncPolicy;

    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), frontLeftSub, frontRightSub, rightSub, leftSub, backSub);
    sync.registerCallback(boost::bind(&pointcloud_callback, _1, _2, _3, _4, _5));

    ros::spin();
}
