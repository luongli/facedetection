#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include <QObject>
#include <stdio.h>
#include <stdlib.h>

// opencv lib
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video/background_segm.hpp"
#include <cv.h>
#include <highgui.h>
#include "opencv2/objdetect.hpp"

#include "my_qlabel.h"
#include "Ctracker.h"

using namespace cv;
using namespace std;

// global variables
CTracker tracker(0.2, 0.5, 60.0, 10, 10); // create a tracker
Scalar Colors[] = {
    Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
    Scalar(255, 255, 0), Scalar(50, 100, 200), Scalar(255, 0, 255),
    Scalar(255, 127, 255), Scalar(127, 0, 255), Scalar(127, 0, 127)
};

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->cameraView, SIGNAL(Mouse_Pressed()), this, SLOT(openCamera()));

    faceCascadeFile = "../facedetection/xml-features/haarcascade_frontalface_default.xml";
    eyesCascadeFile = "../facedetection/xml-features/haarcascade_eye.xml";
    loaded = true;
    camOpened = false;

    // load cascade file
    if( !faceCascade.load( faceCascadeFile ) ){
        cout << "Cannot load file " << faceCascadeFile << endl;
        loaded = false;
    }
    if( !eyesCascade.load( eyesCascadeFile ) ){
        cout << "Cannot load file " << eyesCascadeFile << endl;
        loaded = false;
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::openCamera() {

    if (camOpened) {
        cout << "Camera is being used" << endl;
        return;
    }

    // open camera
    VideoCapture cap(0); // default camera
    if(!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return;
    }

    camOpened = true;
    namedWindow("live video", 1);

    if (loaded) {
        // if cascade files are loaded successfully
        // detect eyes & faces
        detectFaceAndEyes(cap);
    } else {
        showCamera(cap);
    }


    cout << "finished" << endl;
}


void MainWindow::setImage(Mat img, my_qlabel *label){
    Mat img2=img.clone();
    QSize qSize=label->size();
    Size size(qSize.width(),qSize.height());

    if(img2.channels()==3){
        cvtColor(img2,img2,CV_BGR2RGB);
        cv::resize(img2,img2,size);

        QImage qimgOriginal((uchar*)img2.data,img2.cols,img2.rows,img2.step,QImage::Format_RGB888);
        label->setPixmap(QPixmap::fromImage(qimgOriginal));
    }

}

void MainWindow::detectFaceAndEyes(VideoCapture cap) {

    if (!loaded) return;

    Mat frame;
    Mat grayFrame;
    Mat faceROI;
    vector<Rect> faces;
    vector<Rect> eyes;

    // variables used for tracking
    vector<Point2d> centers;

    while(1) {
        cap >> frame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // look for faces in the frame
        faceCascade.detectMultiScale(grayFrame, faces, 1.3);
        centers.clear(); // clear all previous center points
        for( size_t i = 0; i < faces.size(); i++ ) {
            // draw a reactangle bounding each face
            Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
            centers.push_back(center);
            rectangle(frame, faces[i], Scalar(255,0,0), 2);

            // look for eyes in each face
            faceROI = grayFrame(faces[i]);
            eyesCascade.detectMultiScale(faceROI, eyes, 1.3, 2, 0, Size(30, 30));
            for( size_t j = 0; j < eyes.size(); j++ ){
                Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( frame, center, radius, Scalar( 0, 255, 0 ), 4, 8, 0 );
            }
        }

        // draw the trace of trackers
        if (centers.size() > 0) {
            tracker.Update(centers);
            for (int i = 0; i < tracker.tracks.size(); i++) {
                int traceNum = tracker.tracks[i]->trace.size();
                if (traceNum>3){
                    for (int j = 0; j<tracker.tracks[i]->trace.size() - 1; j++){
                        line(frame, tracker.tracks[i]->trace[j], tracker.tracks[i]->trace[j + 1], Colors[tracker.tracks[i]->track_id % 9], 1, CV_AA);
                    }
                    circle(frame, tracker.tracks[i]->trace[traceNum - 1], 2, Colors[tracker.tracks[i]->track_id % 9], 2, 8, 0);
                }
            }
        }

        setImage(frame, ui->cameraView);
        //imshow("live video", frame);
        if(waitKey(30) >= 0) {
            camOpened = false;
            break;
        }
    }
}

void MainWindow::showCamera(VideoCapture cap) {
    Mat frame;

    while(1) {
        cap >> frame;
        imshow("live video", frame);
        setImage(frame, ui->cameraView);
        if(waitKey(30) >= 0) break;
    }
}
