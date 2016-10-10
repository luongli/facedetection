#ifndef MY_LCD_H
#define MY_LCD_H

#include <QWidget>
#include <QLCDNumber>

class my_LCD : public QLCDNumber
{
    Q_OBJECT
public:
    my_LCD();
};

#endif // MY_LCD_H
