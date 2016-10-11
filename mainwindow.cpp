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
CTracker tracker(0.2, 0.5, 200.0, 60, 50); // create a tracker
Scalar Colors[] = {
    Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
    Scalar(255, 255, 0), Scalar(50, 100, 200), Scalar(255, 0, 255),
    Scalar(255, 127, 255), Scalar(127, 0, 255), Scalar(127, 0, 127)
};
int locate = 0;
int position = 0;
QGraphicsRectItem	*	pRect = new QGraphicsRectItem( 0, 0, 0, 0 );
QGraphicsScene* scene;
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QDir dir("/home/hembit/workspace/C++/facedetection/faces/");
    dir.setNameFilters(QStringList() << "*.png" << "*.jpg");
    QStringList fileList = dir.entryList();

    scene = new QGraphicsScene();
    QGraphicsView* view = ui->ImageView;
    pRect->setBrush( Qt::white );
    scene->addItem(pRect);

    QPointF center = view->viewport()->rect().center();
    //center = view->mapToScene(center);

    //qDebug() << fileList;
    foreach (QString path, fileList)
    {

        QPixmap pix("/home/hembit/workspace/C++/facedetection/faces/" + path);
        QGraphicsPixmapItem* item = new QGraphicsPixmapItem(pix.scaled(100,100,Qt::KeepAspectRatio),pRect);
        //scene->addPixmap(pix.scaled(60,60,Qt::KeepAspectRatio));
        item->setParentItem(pRect);
        item->setPos(position*100,0);
        //int pos_x = view->horizontalScrollBar()->value();
        //int pos_y = view->verticalScrollBar()->value();
        scene->addItem(item);
        //view->horizontalScrollBar()->setValue(pos_x);
        //view->verticalScrollBar()->setValue(pos_y);
        position++;
        //QPointF point = itemUnderCursor->mapToScene(itemUnderCursor->boundingRect().topLeft());
        //view->ensureVisible(item);

    }
    //view->ensureVisible(point);
    view->centerOn(center);
    //view->horizontalScrollBar()->setValue( view->horizontalScrollBar()->maximum() );
    //view->horizontalScrollBar()->setValue
    view->setScene(scene);
    view->show();


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
    vcap.release();
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

void MainWindow::openCamera(String source) {

    if (camOpened) {
        cout << "Camera is being used" << endl;
        return;
    }

    if (source.empty()) {
        if (!vcap.open(0)) {
            ui->statusBar->showMessage("Cannot open default camera");
            cout << "cannot open default camera" << endl;
            return;
        }
    } else {
        if(!vcap.open(source)) {
            ui->statusBar->showMessage("Cannot open IP camera. Check your url");
            cout << "Cannot open IP camera. Check your url" << endl;
            return;
        }
    }

    //String videoAddress = "http://ip/mjpg/video.mjpg";
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
        }
        // in case using front camera, flip image around y axis
        flip(original, original, 1);
        frame = original.clone();
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // look for faces in the frame
        faceCascade.detectMultiScale(grayFrame, faces, 1.3, 2, 0, Size(40, 40));
        centers.clear(); // clear all previous center points
        for( size_t i = 0; i < faces.size(); i++ ) {
            if(faces[i].width*faces[i].height<25000 && faces[i].width*faces[i].height>2000){
                // draw a reactangle bounding each face
                Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height );
                centers.push_back(center);
                rectangle(frame, faces[i], Scalar(255,0,0), 2);
            }
        }

        // draw the trace of trackers
        if (centers.size() > 0) {
            tracker.Update(centers, newDetections);
            int traceNum = 0;
            for (int i = 0; i < tracker.tracks.size(); i++) {
                //cout << "assiged Id: " << tracker.tracks[i]->assignedDetectionId << endl;
                if (tracker.tracks[i]->assignedDetectionId < 0) continue;
                traceNum = tracker.tracks[i]->trace.size();
                if (traceNum>20){
                    for (int j = traceNum-10; j<traceNum - 1; j++){
                        line(frame, tracker.tracks[i]->trace[j], tracker.tracks[i]->trace[j + 1], Colors[tracker.tracks[i]->track_id % 9], 1, CV_AA);
                    }
                    circle(frame, tracker.tracks[i]->trace[traceNum - 1], 2, Colors[tracker.tracks[i]->track_id % 9], 2, 8, 0);
                }
                // check if the tracker is captured
                if (!(tracker.tracks[i]->captured) && tracker.tracks[i]->age > 5 && traceNum > 3) {
                    // if the tracker is not captured and it has at least 2 trace points
                    // calculate the distance of the last two traces
                    Point2d diff = tracker.tracks[i]->trace[traceNum-1] - tracker.tracks[i]->trace[traceNum-2];
                    double traceDistance = sqrtf(diff.x*diff.x+diff.y*diff.y);
                    if (traceDistance < dstThreshold) {
                        // if the moving speed is slow
                        // take a picture of that person
                        faceToSave = original(faces[tracker.tracks[i]->assignedDetectionId]);
                        saveFace(faceToSave);
                        rectangle(frame, faces[tracker.tracks[i]->assignedDetectionId], Scalar(rand()%256,rand()%256, rand()%256), -1);
                        tracker.tracks[i]->captured = true;
                        peopleCount++;
                        ui->lcdNumber->display(peopleCount);
                    }
                }
            }
        }

        setImage(frame, ui->cameraView);
        imshow("live video", frame);
        if(waitKey(30) >= 0) {
            vcap.release();
            camOpened = false;
            break;
        }
    }
}

void MainWindow::showCamera() {
    Mat frame;

    while(openning) {
        if(!vcap.read(frame)) {
            cout << "no frame" << endl;
            break;
        }
        // in case using front camera, flip image around y axis
        flip(frame, frame, 1);
        imshow("live video", frame);
        setImage(frame, ui->cameraView);
        if(waitKey(30) >= 0) {
            vcap.release();
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

        QPixmap pix = QPixmap::fromImage(QImage((unsigned char*) faceToSave.data, faceToSave.cols, faceToSave.rows, faceToSave.step, QImage::Format_RGB888));
        QGraphicsView* view = ui->ImageView;
        QGraphicsPixmapItem* item = new QGraphicsPixmapItem(pix.scaled(100,100,Qt::KeepAspectRatio),pRect);

        item->setParentItem(pRect);
        item->setPos(position*100,0);
        scene->addItem(item);
        QPointF center = item->mapToScene(0,0);
        position++;


        if(position > 20) {

//            QGraphicsScene* scene = widget->scene();
//                QList<QGraphicsItem*> items = scene->items();
//                for (int i = 0; i < items.size(); i++) {
//                    scene->removeItem(items[i]);
//                    delete items[i];
//                }
            QGraphicsItem *graphicItem = scene->itemAt(locate*100,0,QTransform());
            scene->removeItem(graphicItem);
            delete graphicItem;

            locate++;
            view->centerOn(center);
            view->setScene(scene);
            view->show();
        } else {
            view->centerOn(center);
            view->setScene(scene);
            view->show();
        }


    } catch (runtime_error& ex) {
        ui->statusBar->showMessage("Exception converting image to PNG format");
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
