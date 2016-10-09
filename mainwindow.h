#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <iostream>
#include <sstream>
#include <fstream>
#include <QObject>
#include <stdio.h>
#include <stdlib.h>
#include <QCloseEvent>
#include <QShowEvent>
#include <QResizeEvent>

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

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    void setImage(Mat img, my_qlabel *label);
    ~MainWindow();

public slots:
    void openCamera();
private:
    Ui::MainWindow *ui;
    String faceCascadeFile;
    String eyesCascadeFile;
    String logoPath = "../facedetection/logo/";
    String facesPath;
    String configPath;
    Mat hustLogo;
    Mat soictLogo;
    Mat celebrateLogo;
    CascadeClassifier faceCascade;
    CascadeClassifier eyesCascade;
    VideoCapture* cap;
    vector<int> compression_params;
    bool loaded;
    bool camOpened;
    bool openning;
    int peopleCount;
    int faceIndex;

    void detectFaceAndEyes();
    void showCamera();
    void closeEvent(QCloseEvent *ev);
    void showEvent(QShowEvent *ev);
    void resizeEvent(QResizeEvent *ev);
    void loadLogos();
    void setLogos();
    void saveFace(Mat faceToSave);
    int loadFaceIndex();
    void saveFaceIndex();
};

#endif // MAINWINDOW_H
