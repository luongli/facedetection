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
#include <QGraphicsScene>

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
private slots:
    //void on_pushButton_clicked();

    //void on_label_linkActivated(const QString &link);

    //void on_imgslide_linkActivated(const QString &link);

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
    double dstThreshold;

    void detectFaceAndEyes(VideoCapture vcap);
    void showCamera();
    void closeEvent(QCloseEvent *ev);
    void showEvent(QShowEvent *ev);
    void resizeEvent(QResizeEvent *ev);
    void loadLogos();
    void setLogos();
    void saveFace(Mat faceToSave, QGraphicsScene * scene, QGraphicsView * view, QGraphicsRectItem * pRect);
    //void saveFace(Mat faceToSave, QGraphicsView * view);
    int loadFaceIndex();
    void saveFaceIndex();
    void clearScene(QGraphicsScene* scene);
    QImage Mat2QImage(cv::Mat const& src);
    void loadImagefromDir(/*QGraphicsScene* scene*/);
    void loadImageToScene(QGraphicsRectItem * pRect, Mat facetoSave, QGraphicsScene * scene, QGraphicsView * view);
};

#endif // MAINWINDOW_H
