
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Time.h>
#include <pcl_conversions/pcl_conversions.h>


typedef sensor_msgs::PointCloud2 pc;
typedef const sensor_msgs::PointCloud2 ppc;

ros::Publisher pcl_pub;

void pointcloud_callback(const boost::shared_ptr<pc const>& frontLeft, const boost::shared_ptr<pc const>&frontRight, const boost::shared_ptr<pc const>&right, const boost::shared_ptr<pc const>&left, const boost::shared_ptr<pc const>& back)
{
    sensor_msgs::PointCloud2 out;
    if (pcl::concatenatePointCloud(*frontLeft, *frontRight, out))
        {
	if(pcl::concatenatePointCloud(out, *right, out))
             if(pcl::concatenatePointCloud(out, *left, out))
                 if(pcl::concatenatePointCloud(out, *back, out))
    // auto summedCloud = *frontLeft + *frontRight + *right + *left + *back;
                    pcl_pub.publish(out);
	}
	else{
std::cout << "cannot concat fleft, fright \n\n";
}

    // pcl::conversions::concatenatePointCloud()
}


int main( int argc, char** argv)
{
    ros::init(argc, argv, "concatenate_pointclouds");
    ros::NodeHandle nh;
    pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("concatenated_pointcloud", 1);

    message_filters::Subscriber<sensor_msgs::PointCloud2> frontLeftSub(nh, "/pointcloud/frontLeft", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> frontRightSub(nh, "/pointcloud/frontRight", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> rightSub(nh, "/pointcloud/right", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> leftSub(nh, "/pointcloud/left", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> backSub(nh, "/pointcloud/back", 1);
    message_filters::TimeSynchronizer<pc, pc, pc, pc, pc> sync(frontLeftSub, frontRightSub, rightSub, leftSub, backSub, 5);
    sync.registerCallback(pointcloud_callback);

    ros::spin();
    return 0;
}
