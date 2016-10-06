#ifndef MY_QLABEL_H
#define MY_QLABEL_H

#include <QLabel>
#include <QMouseEvent>
#include <QEvent>
#include <QDebug>
#include <stdio.h>

class my_qlabel : public QLabel
{
    Q_OBJECT
public:
    explicit my_qlabel(QWidget *parent = 0);
    void mousePressEvent(QMouseEvent *ev);

signals:
    void Mouse_Pressed();
};

#endif // MY_QLABEL_H
