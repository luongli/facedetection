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
#include <QInputDialog>
#include <QLabel>

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
CTracker tracker(0.2, 0.5, 100.0, 30, 40); // create a tracker
Scalar Colors[] = {
    Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
    Scalar(255, 255, 0), Scalar(50, 100, 200), Scalar(255, 0, 255),
    Scalar(255, 127, 255), Scalar(127, 0, 255), Scalar(127, 0, 127)
};
int locate = 0;             // delete item at position locate
int position = 0;           // track position of new item
int constraint = 200;       // max item in view/scene
QGraphicsRectItem	*	pRect  =  new QGraphicsRectItem( 0, 0, 0, 0 );
QGraphicsScene* scene;
QGraphicsView* view;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle("Check In Here");

    connect(ui->cameraView, SIGNAL(Mouse_Pressed()), this, SLOT(openCamera()));
    connect(ui->actionDefault_camera, SIGNAL(triggered(bool)), this, SLOT(openCamera()));
    connect(ui->actionIP_camera, SIGNAL(triggered(bool)), this, SLOT(openIpCamera()));
    faceCascadeFile = "../facedetection/xml-features/haarcascade_frontalface_alt.xml";
    eyesCascadeFile = "../facedetection/xml-features/haarcascade_eye.xml";
    facesPath = "../facedetection/faces/";
    configPath = "../facedetection/config/";
    loadLogos();
    loaded = true;
    camOpened = false;
    openning = true;
    peopleCount = loadCountIndex();
    faceIndex = loadFaceIndex();
    dstThreshold = 5;
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
    ui->lcdNumber->display(peopleCount);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *ev) {
    cout << "closing" << endl;
    saveFaceIndex();
    saveCountIndex();
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

    if (celebrateBg.data) {
        setImage(celebrateBg, ui->cameraView);
        setImage(celebrateLogo, ui->celebrateLogo);
    } else {
        cout << "celebrate background is not loaded" << endl;
    }

    if (celebrateLogo.data) {
        setImage(celebrateLogo, ui->celebrateLogo);
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
    scene = new QGraphicsScene();
    view = ui->ImageView;
    pRect->setBrush( Qt::white );
    scene->addItem(pRect);

    while(openning) {
        if(!vcap.read(original)) {
            cout << "no frame" << endl;
            break;
        }
        cv::resize(original, original, Size(640, 400));
        // in case using front camera, flip image around y axis
        flip(original, original, 1);
        frame = original.clone();
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // look for faces in the frame
        faceCascade.detectMultiScale(grayFrame, faces, 1.2, 2, 0, Size(20, 20));
        centers.clear(); // clear all previous center points
        for( size_t i = 0; i < faces.size(); i++ ) {
            if(faces[i].width*faces[i].height>400){
                // draw a reactangle bounding each face
                Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height );
                centers.push_back(center);
                rectangle(frame, faces[i], Scalar(255,0,0), 2);
                //cout<<faces[i].x<<" "<<faces[i].y<<" "<<faces[i].height*faces[i].width<<endl;
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
                // check if the tracker is captured  && tracker.tracks[i]->age > 5 && traceNum > 3
                if (!(tracker.tracks[i]->captured)) {
                    // if the tracker is not captured and it has at least 2 trace points
                    // calculate the distance of the last two traces
                    faceToSave = original(faces[tracker.tracks[i]->assignedDetectionId]);
                    bool isFace = recheckFace(&faceToSave);
                    //cout << "isFace = " << isFace << endl;

                    if (isFace) {
                        tracker.tracks[i]->trueCount++;
                    }

                    if (tracker.tracks[i]->age >= 10) {
                        if (tracker.tracks[i]->trueCount >= 7) {
                            // test the face for 10 times
                            // if it's 7 out of 10 detected as a face, save it
                            saveFace(faceToSave,scene,view, pRect);
                            rectangle(frame, faces[tracker.tracks[i]->assignedDetectionId], Scalar(rand()%256,rand()%256, rand()%256), -1);
                            tracker.tracks[i]->captured = true;
                            peopleCount++;
                            ui->lcdNumber->display(peopleCount);
                        } else {
                            // otherwise we ignore it
                            tracker.tracks[i]->captured = true;
                        }
                    }
                }
            }
        }

        setImage(frame, ui->cameraView);
        //imshow("live video", frame);
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
    soictLogo = imread(logoPath + "soict.jpg", CV_LOAD_IMAGE_COLOR);
    celebrateLogo = imread(logoPath + "logo60.png", CV_LOAD_IMAGE_COLOR);
    celebrateBg = imread(logoPath + "logo60bg.png", CV_LOAD_IMAGE_COLOR);
}


/**********************************************************************
 * This function will save captured face into the disk at faces folder
 * and put the preview of that faces into the top qlistview
 * *******************************************************************/
void MainWindow::saveFace(Mat faceToSave, QGraphicsScene * scene, QGraphicsView * view, QGraphicsRectItem * pRect) {
    stringstream ss;
    ss << faceIndex;
    String fileName = ss.str();
    try {
        imwrite(facesPath + fileName + ".png", faceToSave, compression_params);
        faceIndex++;
        position++;
        if((position % 200)==199) {
            scene->clear();
            QGraphicsRectItem	*	pRect  =  new QGraphicsRectItem( 0, 0, 0, 0 );
            view->viewport()->update();
        } else {
            QImage qimgOriginal = Mat2QImage(faceToSave);
            //QImage qimgOriginal((uchar*)faceToSave.data,faceToSave.cols,faceToSave.rows,faceToSave.step,QImage::Format_RGB888);
            QPixmap pix = QPixmap::fromImage(qimgOriginal,Qt::ColorMode_Mask);
            QGraphicsPixmapItem* item = new QGraphicsPixmapItem(pix.scaled(100,100,Qt::KeepAspectRatio),pRect);

            item->setParentItem(pRect);
            item->setPos(position*100,0);
            scene->addItem(item);

            view->setScene(scene);
            QPointF center = item->mapToScene(0,0);
            view->centerOn(center);
            view->show();
            qDebug() << scene->items().count();
            qDebug() << view->items().count();
        }


    } catch (runtime_error& ex) {
        ui->statusBar->showMessage("Exception converting image to PNG format");
    }
}

/*****************************************************************
 * Load item to Scene
 *****************************************************************/
void MainWindow::loadImageToScene(QGraphicsRectItem * pRect, Mat faceToSave, QGraphicsScene * scene, QGraphicsView * view)
{
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
    qDebug() << scene->items().count();
    qDebug() << view->items().count();
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
}

QImage MainWindow::Mat2QImage(cv::Mat const& src)
{
     cv::Mat temp; // make the same cv::Mat
     cvtColor(src, temp,CV_BGR2RGB); // cvtColor Makes a copt, that what i need
     QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
     dest.bits(); // enforce deep copy, see documentation
     // of QImage::QImage ( const uchar * data, int width, int height, Format format )
     return dest;
}

/*****************************************************
 * This function controls how many item is cleared
 *****************************************************/
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

int MainWindow::loadCountIndex() {
    ui->statusBar->showMessage("Loading face index...");
    ifstream input((configPath + "count.conf").c_str(), ios::in);
    int index = 0;
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

void MainWindow::saveCountIndex() {
    ui->statusBar->showMessage("Saving face index...");
    ofstream output((configPath + "count.conf").c_str(), ios::out|ios::trunc);
    if(output.is_open()) {
        // if the file is open successfully

        output << peopleCount << "\n";
        output.close();
        ui->statusBar->showMessage("Face index saved", 1000);
    } else {
        cout << "cannot open file" << endl;
        ui->statusBar->showMessage("Error: Cannot save face index", 1000);
    }
}


void MainWindow::openIpCamera() {
    // ask for video link
    bool ok;
    QString text = QInputDialog::getText(this, tr("Video Stream Adress"),
                                         tr("Input video stream address including username and password of the ip camera"), QLineEdit::Normal,
                                         QDir::home().dirName(), &ok);
    if (ok && !text.isEmpty()) {
        String address = text.toStdString();
        openCamera(address);
    }
}

bool MainWindow::recheckFace(Mat *face){
    vector<Rect> faces;

    Size s = face->size();

    faceCascade.detectMultiScale(*face, faces, 1.2, 2, 0, Size(s.width/1.5, s.height/1.5));

    if (faces.size() <= 0) {
        return false;
    }

    return true;
}
