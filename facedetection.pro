#-------------------------------------------------
#
# Project created by QtCreator 2016-10-05T17:29:35
#
#-------------------------------------------------

QT       += core gui
QT += printsupport
CONFIG += debug

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = facedetection
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    my_qlabel.cpp \
    Kalman.cpp \
    HungarianAlg.cpp \
    Ctracker.cpp

HEADERS  += mainwindow.h \
    my_qlabel.h \
    Kalman.h \
    HungarianAlg.h \
    Ctracker.h

FORMS    += mainwindow.ui

INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib \
-lopencv_calib3d \
-lopencv_contrib \
-lopencv_core \
-lopencv_features2d \
-lopencv_flann \
-lopencv_gpu \
-lopencv_highgui \
-lopencv_legacy \
-lopencv_ml \
-lopencv_objdetect \
-lopencv_ocl \
-lopencv_photo \
-lopencv_stitching \
-lopencv_imgcodecs \
-lopencv_imgproc \
-lopencv_superres \
-lopencv_video \
-lopencv_videoio \
-lopencv_videostab \
-lopencv_viz \

