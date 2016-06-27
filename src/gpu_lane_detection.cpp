#include <cmath>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define DEBUG

struct Lane {
	Lane(){}
	Lane(Point p0, Point p1, float angle): p0(p0), p1(p1), angle(angle) { }

	Point p0, p1;
	float angle;
};

enum{
    SCAN_STEP = 5,			  // in pixels
	LEFT_MAX_LINE_REJECT_DEGREES = -8, // in degrees
	LEFT_MIN_LINE_REJECT_DEGREES = -50, // in degrees
	RIGHT_MAX_LINE_REJECT_DEGREES = 50, // in degrees
	RIGHT_MIN_LINE_REJECT_DEGREES = 8, // in degrees
	
	CANNY_MIN_TRESHOLD = 120,	  // edge detector minimum hysteresis threshold
	CANNY_MAX_TRESHOLD = 200, // edge detector maximum hysteresis threshold

	HOUGH_TRESHOLD = 50,		// line approval vote threshold
	HOUGH_MIN_LINE_LENGTH = 15,	// remove lines shorter than this treshold
	HOUGH_MAX_LINE_GAP = 150,   // join lines to one with smaller than this gaps

	FRAME_WAIT_KEY = 1,
	CAR_MIDDLE = 1050,

	LANE_COUNT_THRESHOLD = 0,
	RESULT_SIZE = 55,
};

#define LEFT 1
#define MIDDLE 2
#define RIGHT 3
#define NONE 4


int gpu_processLanes(vector<Vec4i> &lines, Mat &temp_frame, Mat &frame, int* result, int size, int &current) {

	// classify lines to left/right side
	vector<Lane> left, right;
	
	for (size_t i = 0; i < lines.size(); ++i)
	{
		Vec4i line = lines[i];
		int midx = (line[0] + line[2]) / 2;
		int dx = line[2] - line[0];
		int dy = line[3] - line[1];
		float angle = atan2f(dy, dx) * 180/CV_PI;

		// filter noisy
		if(midx < CAR_MIDDLE){
			if((angle > 135 && angle < 152) || (angle > -12 && angle < -6 ))
				left.push_back(Lane(Point(line[0], line[1]), Point(line[2], line[3]), angle));
			else
				continue;
		}
		else if(midx >= CAR_MIDDLE){
			if(angle < 42 && angle > 6)
				right.push_back(Lane(Point(line[0], line[1]), Point(line[2], line[3]), angle));
			else
				continue;
		}
		/*
		// print lanes angle
		char ee[100];
		sprintf(ee, "%d", (int)angle);
		putText(temp_frame, ee, Point(int((line[0]+line[2])/2), int((line[1]+line[3])/2)), CV_FONT_VECTOR0, 1, Scalar(255, 178, 0));
		cv::line(temp_frame, Point(line[0], line[1]), Point(line[2], line[3]), cvScalar(0, 0, 255), 3, CV_AA);
		*/

	}
	
	// show Hough lines
	for	(int i=0; i<right.size(); i++) {
		Point p0, p1;
		p0.x = right[i].p0.x;
		p0.y = right[i].p0.y + 1080*2.5/3.5;
		p1.x = right[i].p1.x;
		p1.y = right[i].p1.y + 1080*2.5/3.5;
		line(frame, p0, p1, CV_RGB(0, 0, 255), 2);
	}

	for	(int i=0; i<left.size(); i++) {
		Point p0, p1;
		p0.x = left[i].p0.x;
		p0.y = left[i].p0.y + 1080*2.5/3.5;
		p1.x = left[i].p1.x;
		p1.y = left[i].p1.y + 1080*2.5/3.5;
		line(frame, p0, p1, CV_RGB(0, 0, 255), 2);
	}

	// compute right-side left-side lanes
	int lc = 0, rc = 0;
	for(int g=0; g<right.size(); g++){
		if(right[g].angle > 6 && right[g].angle < 12)
			rc++;

	}
	for(int g=0; g<left.size(); g++){
		if(left[g].angle > -12 && left[g].angle < -6)
			lc++;
	}

	// determine the lane belong
	if(lc > LANE_COUNT_THRESHOLD && rc > LANE_COUNT_THRESHOLD){
		result[current] = MIDDLE;
	}
	else if(lc > LANE_COUNT_THRESHOLD){
		result[current] = RIGHT;
	}
	else if(rc > LANE_COUNT_THRESHOLD){
		result[current] = LEFT;
	}
	else{
		result[current] = NONE;
	}
	current = (current+1)%RESULT_SIZE;

	int temp[5] = {0};
	for(int i=0; i<RESULT_SIZE; i++){
		if(result[i] == LEFT)	temp[1]++;
		else if(result[i] == MIDDLE)	temp[2]++;
		else if(result[i] == RIGHT)		temp[3]++;
		else if(result[i] == NONE)	temp[4]++;
		//printf("%d ", result[i]);
	}
	//printf("\ntemp = %d %d %d %d\n", temp[1], temp[2], temp[3], temp[4]);
	int win = NONE, maxc = 0;
	for(int i=1; i<=4; i++){
		if(temp[i] > maxc){
			win = i;
			maxc = temp[i];
		}
	}

	return win;
}


int gpu_lane_detection(Mat input, int* result, int size, int &current){

	Size video_size = Size(1920, 1080);
	Size temp_frame_size = Size(1920, 1080/3.5 - 70);
	Mat temp_frame = Mat(temp_frame_size, CV_8UC3);
	Mat grey = Mat(temp_frame_size, CV_8UC1);
	Mat edges = Mat(temp_frame_size, CV_8UC1);
	
	GpuMat d_temp_frame(temp_frame_size, CV_8UC3);
	GpuMat d_grey(temp_frame_size, CV_8U);
	GpuMat d_edges(temp_frame_size, CV_8U);

	GpuMat d_lines;
	HoughLinesBuf d_buf;

	// gpu do every opencv (expect processLane)
	Rect roi = Rect(0, video_size.height - temp_frame_size.height - 70, temp_frame_size.width, temp_frame_size.height);
	temp_frame = input(roi);

	// gpu do cvtColor + Canny + Hough
	GpuMat Gaussian_buf;
	d_temp_frame.upload(temp_frame);

	gpu::cvtColor(d_temp_frame, d_grey, CV_BGR2GRAY);
	gpu::GaussianBlur(d_grey, d_grey, Size(5, 5), Gaussian_buf, 0);
	gpu::Canny(d_grey, d_edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

	gpu::HoughLinesP(d_edges, d_lines, d_buf, 1, CV_PI / 180, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

	vector<Vec4i> lines_gpu;
	if (!d_lines.empty())
	{
		lines_gpu.resize(d_lines.cols);
		Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
		d_lines.download(h_lines);
	}
	//cout << lines_gpu.size() << endl;

	int out = gpu_processLanes(lines_gpu, temp_frame, input, result, size, current);	

	return out;
}

