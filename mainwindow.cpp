#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include <QObject>
#include <stdio.h>
#include <stdlib.h>
#include <QPixmap>
#include <QFileDialog>
#include <QCloseEvent>
#include <QShowEvent>
#include <QGraphicsPixmapItem>
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

// global variables
CTracker tracker(0.2, 0.5, 60.0, 10, 10); // create a tracker
Scalar Colors[] = {
    Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
    Scalar(255, 255, 0), Scalar(50, 100, 200), Scalar(255, 0, 255),
    Scalar(255, 127, 255), Scalar(127, 0, 255), Scalar(127, 0, 127)
};
int locate = 0;             // delete item at position locate
int position = 0;           // track position of new item
int constraint = 200;       // max item in view/scene
QGraphicsRectItem	*	pRect = new QGraphicsRectItem( 0, 0, 0, 0 );
QGraphicsScene* scene;
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->cameraView, SIGNAL(Mouse_Pressed()), this, SLOT(openCamera()));

    faceCascadeFile = "../facedetection/xml-features/haarcascade_frontalface_default.xml";
    eyesCascadeFile = "../facedetection/xml-features/haarcascade_eye.xml";
    facesPath = "../facedetection/faces/";
    configPath = "../facedetection/config/";
    loadLogos();
    loaded = true;
    camOpened = false;
    openning = true;
    peopleCount = 0;
    faceIndex = loadFaceIndex();
    dstThreshold = 3;
    // params to save images
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(5);

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
    saveFaceIndex();
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

    String videoAddress = "http://ip/mjpg/video.mjpg";
    VideoCapture vcap(0);
    // open camera
    //cap = new VideoCapture(0); // default camera
//    if(!vcap.open(videoAddress)) {
//        cout << "Cannot open camera" << endl;
//        return;
//    } else {
//        cout << "open camera successfully" << endl;
//    }

    camOpened = true;
    namedWindow("live video", 1);

    if (loaded) {
        // if cascade files are loaded successfully
        // detect eyes & faces
        detectFaceAndEyes(vcap);
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

void MainWindow::detectFaceAndEyes(VideoCapture vcap) {

    if (!loaded) return;

    Mat original;
    Mat frame;
    Mat grayFrame;
    Mat faceROI;
    Mat faceToSave;
    vector<Rect> faces;
    vector<Rect> eyes;

    // variables used for tracking
    vector<Point2d> centers;
    vector<int> newDetections;

    while(openning) {
        if(!vcap.read(original)) {
            cout << "no frame" << endl;
            break;
        } else {
            //cout << "read camera successfully" << endl;
        }
        // in case using front camera, flip image around y axis
        flip(original, original, 1);
        frame = original.clone();
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
//            faceROI = grayFrame(faces[i]);
//            eyesCascade.detectMultiScale(faceROI, eyes, 1.3, 2, 0, Size(30, 30));
//            for( size_t j = 0; j < eyes.size(); j++ ){
//                Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
//                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//                circle( frame, center, radius, Scalar( 0, 255, 0 ), 4, 8, 0 );
//            }
        }

        // draw the trace of trackers
        if (centers.size() > 0) {
            tracker.Update(centers, newDetections);
            int traceNum = 0;
            for (int i = 0; i < tracker.tracks.size(); i++) {
                traceNum = tracker.tracks[i]->trace.size();
                if (traceNum>3){
                    for (int j = 0; j<tracker.tracks[i]->trace.size() - 1; j++){
                        line(frame, tracker.tracks[i]->trace[j], tracker.tracks[i]->trace[j + 1], Colors[tracker.tracks[i]->track_id % 9], 1, CV_AA);
                    }
                    circle(frame, tracker.tracks[i]->trace[traceNum - 1], 2, Colors[tracker.tracks[i]->track_id % 9], 2, 8, 0);
                }
                // check if the tracker is captured
                if (!(tracker.tracks[i]->captured) && tracker.tracks[i]->age > 5 && traceNum > 1) {
                    // if the tracker is not captured and it has at least 2 trace points
                    // calculate the distance of the last two traces
                    Point2d diff = tracker.tracks[i]->trace[traceNum-1] - tracker.tracks[i]->trace[traceNum-2];
                    double traceDistance = sqrtf(diff.x*diff.x+diff.y*diff.y);
                    if (traceDistance < dstThreshold) {
                        faceToSave = original(faces[tracker.tracks[i]->assignedDetectionId]);
                        saveFace(faceToSave);
                        rectangle(frame, faces[tracker.tracks[i]->assignedDetectionId], Scalar(rand()%256,rand()%256, rand()%256), -1);
                        peopleCount++;
                        ui->lcdNumber->display(peopleCount);
                        tracker.tracks[i]->captured = true;
                    }
                }
            }

//            if(newDetections.size() > 0) {
//                // if we have new faces
//                // save those faces
//                for (int i = 0; i < newDetections.size(); i++) {
//                    faceToSave = original(faces[newDetections[i]]);
//                    saveFace(faceToSave);
//                    rectangle(frame, faces[newDetections[i]], Scalar(rand()%256,rand()%256, rand()%256), -1);
//                    peopleCount++;
//                    ui->lcdNumber->display(peopleCount);
//                }
//            }
        }

        setImage(frame, ui->cameraView);
        imshow("live video", frame);
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


/**********************************************************************
 * This function will save captured face into the disk at faces folder
 * and put the preview of that faces into the top qlistview
 * *******************************************************************/
void MainWindow::saveFace(Mat faceToSave) {
    stringstream ss;
    ss << faceIndex;
    String fileName = ss.str();
    try {
        imwrite(facesPath + fileName + ".png", faceToSave, compression_params);
        faceIndex++;
        scene = new QGraphicsScene();
        QGraphicsView* view = ui->ImageView;
//        if((position % 10)==5) {
//            clearScene(scene);
//               view->items().clear();
//               view->resetCachedContent();
//               view->updateGeometry();
//               view->repaint();
//               view->viewport()->update();
//               view->show();
////            clearScene(scene);
////            scene->clear();
////            view->acceptDrops()
////            view->viewport()->update();
////            view->setScene(scene);
////            view->show();
//            position++;

////            //loadImagefromDir();
//        } else {
//            cout << "In anh moi ";
//            cout << position << endl;
            pRect->setBrush( Qt::white );
            scene->addItem(pRect);

            //scene = new QGraphicsScene();

            //QGraphicsView* view = ui->ImageView;
            QImage qimgOriginal((uchar*)faceToSave.data,faceToSave.cols,faceToSave.rows,faceToSave.step,QImage::Format_RGB888);

            //QPixmap pix = QPixmap::fromImage(QImage((unsigned char*) rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888));
            QPixmap pix = QPixmap::fromImage(qimgOriginal,Qt::AutoColor);
            QGraphicsPixmapItem* item = new QGraphicsPixmapItem(pix.scaled(100,100,Qt::KeepAspectRatio),pRect);

            item->setParentItem(pRect);
            item->setPos(position*100,0);
            scene->addItem(item);

            view->setScene(scene);
            QPointF center = item->mapToScene(0,0);
            view->centerOn(center);
            view->show();
            position++;
//        }


    } catch (runtime_error& ex) {
        ui->statusBar->showMessage("Exception converting image to PNG format");
    }
}

/*****************************************************************
 * Load image from dir
 *****************************************************************/
void MainWindow::loadImagefromDir(/*QGraphicsScene* scene*/)
{
        position = 0;
        QDir dir("../facedetection/faces/");
        dir.setNameFilters(QStringList() << "*.png" << "*.jpg");
        QStringList fileList = dir.entryList();

        scene = new QGraphicsScene();
        QGraphicsView* view = ui->ImageView;
        pRect->setBrush( Qt::white );
        scene->addItem(pRect);

        //QPointF center = view->viewport()->rect().center();
        for (int i = fileList.length()-5; i < fileList.length(); i++)
        {
            QPixmap pix("../facedetection/faces/" + fileList[i]);
            QGraphicsPixmapItem* item = new QGraphicsPixmapItem(pix.scaled(100,100,Qt::KeepAspectRatio),pRect);
            item->setParentItem(pRect);
            item->setPos(position*100,0);
            scene->addItem(item);
            view->setScene(scene);


            QPointF center = item->mapToScene(0,0);
            view->centerOn(center);
            view->show();
            position++;
        }
        cout << "After add dir ";
        cout << position << endl;
//        foreach (QString path, fileList)
//        {
//            QPixmap pix("../facedetection/faces/" + path);
//            QGraphicsPixmapItem* item = new QGraphicsPixmapItem(pix.scaled(100,100,Qt::KeepAspectRatio),pRect);
//            item->setParentItem(pRect);
//            item->setPos(position*100,0);
//            scene->addItem(item);
//            position++;
//        }



}

void MainWindow::clearScene(QGraphicsScene* scene)
{
  QList<QGraphicsItem*> itemsList = scene->items();
  QList<QGraphicsItem*>::iterator iter = itemsList.begin();
  QList<QGraphicsItem*>::iterator end = itemsList.end();
  while(iter != end)
    {
      QGraphicsItem* item = (*iter);
      scene->removeItem(item);
      delete item;
      iter++;
    }
}

int MainWindow::loadFaceIndex() {
    ui->statusBar->showMessage("Loading face index...");
    ifstream input((configPath + "face-index.conf").c_str(), ios::in);
    int index = -1;
    if(input.is_open()) {
        input >> index;
        input.close();
        ui->statusBar->clearMessage();
        return index;
    } else {
        cout << "Be careful! face index is not loaded successfully" << endl;
        ui->statusBar->showMessage("Error: face index is not loaded successfully", 1000);
        return 0;
    }
}


void MainWindow::saveFaceIndex() {
    ui->statusBar->showMessage("Saving face index...");
    ofstream output((configPath + "face-index.conf").c_str(), ios::out|ios::trunc);
    if(output.is_open()) {
        // if the file is open successfully

        output << faceIndex << "\n";
        output.close();
        ui->statusBar->showMessage("Face index saved", 1000);
    } else {
        cout << "cannot open file" << endl;
        ui->statusBar->showMessage("Error: Cannot save face index", 1000);
    }
}
