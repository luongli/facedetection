#pragma once
#include "Kalman.h"
#include "HungarianAlg.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

class CTrack
{
public:
    vector<Point2d> trace;
    static size_t NextTrackID;
    bool captured;
    size_t track_id;
    size_t skipped_frames;
    size_t crossBorder;
    Point2d prediction;
    int assignedDetectionId;
    int age;
    int trueCount;
    TKalmanFilter* KF;
    CTrack(Point2f p, float dt, float Accel_noise_mag);
    ~CTrack();
};


class CTracker
{
public:

    float dt;
    float Accel_noise_mag;
    double dist_thres;
    int maximum_allowed_skipped_frames;
    int max_trace_length;

    vector<CTrack*> tracks;
    void Update(vector<Point2d>& detections, vector<int>& newDetections);
    CTracker(float _dt, float _Accel_noise_mag, double _dist_thres=60, int _maximum_allowed_skipped_frames=10,int _max_trace_length=10);
    ~CTracker(void);
};

