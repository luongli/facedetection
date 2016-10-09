#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include <QObject>
#include <stdio.h>
#include <stdlib.h>
#include <QCloseEvent>
#include <QShowEvent>

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
    loadLogos();
    loaded = true;
    camOpened = false;
    openning = true;
    peopleCount = 0;

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

void MainWindow::closeEvent(QCloseEvent *ev) {
    cout << "closing" << endl;
    openning = false;
    cap->release();
    delete cap;
    destroyAllWindows();
}

void MainWindow::showEvent(QShowEvent *ev) {
    setLogos();
}

void MainWindow::resizeEvent(QResizeEvent *ev) {
    setLogos();
}

void MainWindow::setLogos() {
    // display logos
    if (hustLogo.data) {
        setImage(hustLogo, ui->hustLogo);
    } else {
        cout << "Hust logo is not loaded " << endl;
    }

    if (soictLogo.data) {
        setImage(soictLogo, ui->soictLogo);
    } else {
        cout << "Soict logo is not loaded " << endl;
    }

    if (celebrateLogo.data) {
        setImage(celebrateLogo, ui->cameraView);
    } else {
        cout << "celebrate logo is not loaded" << endl;
    }
}

void MainWindow::openCamera() {

    if (camOpened) {
        cout << "Camera is being used" << endl;
        return;
    }

    // open camera
    cap = new VideoCapture(0); // default camera
    if(!cap->isOpened()) {
        cout << "Cannot open camera" << endl;
        return;
    }

    camOpened = true;
    namedWindow("live video", 1);

    if (loaded) {
        // if cascade files are loaded successfully
        // detect eyes & faces
        detectFaceAndEyes();
    } else {
        showCamera();
    }


    cout << "paused" << endl;
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

void MainWindow::detectFaceAndEyes() {

    if (!loaded) return;

    Mat frame;
    Mat grayFrame;
    Mat faceROI;
    vector<Rect> faces;
    vector<Rect> eyes;

    // variables used for tracking
    vector<Point2d> centers;
    vector<int> newDetections;

    while(openning) {
        *cap >> frame;
        // in case using front camera, flip image around y axis
        flip(frame, frame, 1);
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // look for faces in the frame
        faceCascade.detectMultiScale(grayFrame, faces, 1.3, 2, 0, Size(30, 30));
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
            tracker.Update(centers, newDetections);
            for (int i = 0; i < tracker.tracks.size(); i++) {
                int traceNum = tracker.tracks[i]->trace.size();
                if (traceNum>3){
                    for (int j = 0; j<tracker.tracks[i]->trace.size() - 1; j++){
                        line(frame, tracker.tracks[i]->trace[j], tracker.tracks[i]->trace[j + 1], Colors[tracker.tracks[i]->track_id % 9], 1, CV_AA);
                    }
                    circle(frame, tracker.tracks[i]->trace[traceNum - 1], 2, Colors[tracker.tracks[i]->track_id % 9], 2, 8, 0);
                }
            }

            if(newDetections.size() > 0) {
                // if we have new faces
                // save those faces
                for (int i = 0; i < newDetections.size(); i++) {
                    rectangle(frame, faces[newDetections[i]], Scalar(rand()%256,rand()%256, rand()%256), -1);
                    peopleCount++;
                }
            }
        }

        setImage(frame, ui->cameraView);
        //imshow("live video", frame);
        if(waitKey(30) >= 0) {
            cap->release();
            camOpened = false;
            break;
        }
    }
}

void MainWindow::showCamera() {
    Mat frame;

    while(openning) {
        *cap >> frame;
        // in case using front camera, flip image around y axis
        flip(frame, frame, 1);
        imshow("live video", frame);
        setImage(frame, ui->cameraView);
        if(waitKey(30) >= 0) {
            cap->release();
            camOpened = false;
            break;
        }
    }
}


void MainWindow::loadLogos() {
    hustLogo = imread(logoPath + "hust.png", CV_LOAD_IMAGE_COLOR);
    soictLogo = imread(logoPath + "soict.png", CV_LOAD_IMAGE_COLOR);
    celebrateLogo = imread(logoPath + "logo60.png", CV_LOAD_IMAGE_COLOR);
}
