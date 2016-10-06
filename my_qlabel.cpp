#include "my_qlabel.h"
#include <stdio.h>
#include <iostream>

using namespace std;

my_qlabel::my_qlabel(QWidget *parent) : QLabel(parent)
{
}

void my_qlabel::mousePressEvent(QMouseEvent *ev) {
    cout << "ok" << endl;
}
