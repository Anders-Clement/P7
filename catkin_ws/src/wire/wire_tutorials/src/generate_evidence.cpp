/*
 * generate_evidence.cpp
 *
 *  Created on: Nov 12, 2012
 *      Author: jelfring
 */

#include <ros/ros.h>

#include "wire_msgs/WorldEvidence.h"
#include "wire_msgs/ObjectEvidence.h"
#include "geometry_msgs/QuaternionStamped.h"
#include "problib/conversions.h"


ros::Publisher world_evidence_publisher_;

void addEvidence(wire_msgs::WorldEvidence& world_evidence, _Float64 x, _Float64 y, _Float64 z, _Float64 confidence) {
	wire_msgs::ObjectEvidence obj_evidence;

	// Set the continuous position property
	wire_msgs::Property posProp;
	posProp.attribute = "position";

	// Set position (x,y,z), set the covariance matrix as 0.005*identity_matrix
	double cov = .025;
	pbl::PDFtoMsg(pbl::Gaussian(pbl::Vector3(x, y, z), pbl::Matrix3(cov, cov, cov)), posProp.pdf);
	obj_evidence.properties.push_back(posProp);
	
	

	// Set the discrete class label property
    wire_msgs::Property classProp;
    classProp.attribute = "class_label";
    pbl::PMF classPMF;
	
    // Probability of the class label
    classPMF.setProbability("human", confidence);
    pbl::PDFtoMsg(classPMF, classProp.pdf);
    obj_evidence.properties.push_back(classProp);
	world_evidence.object_evidence.push_back(obj_evidence);
}


// void generateEvidence() {

// 	// Create world evidence message
// 	wire_msgs::WorldEvidence world_evidence;

// 	// Set header
// 	world_evidence.header.stamp = ros::Time::now();
// 	world_evidence.header.frame_id = "/map";

// 	// Add evidence
// 	addEvidence(world_evidence,1,1,1,0.7,"human");

// 	// Publish results
// 	world_evidence_publisher_.publish(world_evidence);
// 	ROS_INFO("Published world evidence with size %d", world_evidence.object_evidence.size());

// }

void evidenceCallback(const geometry_msgs::QuaternionStamped::ConstPtr msg) // CHANGE TO MESSAGE TYPE WHEN MADE
{
  	wire_msgs::WorldEvidence world_evidence;

	// Set header
	world_evidence.header = msg->header;

	// Add evidence
	auto q = msg->quaternion;
	addEvidence(world_evidence, q.x, q.y, q.z, q.w);

	world_evidence_publisher_.publish(world_evidence);
	
	
	ROS_INFO("Published world evidence with size %d", world_evidence.object_evidence.size());

}

/**
 * Main
 */
int main(int argc, char **argv) {

	// Initialize ros and create node handle
	ros::init(argc,argv,"generate_evidence");
	ros::NodeHandle nh;


	// Publisher used to send evidence to world model
	
	world_evidence_publisher_ = nh.advertise<wire_msgs::WorldEvidence>("/world_evidence", 100);

	//Subscriber
	ros::Subscriber sub = nh.subscribe("/people_detections", 1000, evidenceCallback);
	
	
	// // Publish with 3 Hz
	// //ros::Rate r(3.0);

	// //while (ros::ok()) {
	// 	generateEvidence();
	// 	r.sleep();
	// }

	ros::spin();

	return 0;
}
