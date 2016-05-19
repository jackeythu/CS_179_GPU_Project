#include <math.h>
#include "Point2d.h"

//constructor, destructor and assignment operator
Point2d::Point2d(double x, double y) : x_(x), y_(y) {}

//scaler multiplication and division
Point2d& Point2d::operator*=(double c)
{
	x_ *= c;
	y_ *= c;
	return *this;
}

Point2d& Point2d::operator/=(double c)
{
	x_ /= c;
	y_ /= c;
	return *this;
}

//vector addition and subtraction
Point2d& Point2d::operator+=(const Point2d& u)
{
	x_ += u.x();
	y_ += u.y();
	return *this;
}

Point2d& Point2d::operator-=(const Point2d& u)
{
	x_ -= u.x();
	y_ -= u.y();
	return *this;
}

//subscript: index 0 and 1 returns reference to x_ and y_

//........


//outclass operators
Point2d operator*(const Point2d& u, double c)
{
	Point2d result(u);
	result *= c;
	return result;
}

Point2d operator*(double c, const Point2d& u)
{
	return u * c;
}

double operator*(const Point2d& u1, const Point2d& u2)
{
	return u1.x()*u2.x() + u1.y()*u2.y();
}

Point2d operator/(const Point2d& u, double c)
{
	Point2d result(u);
	result /= c;
	return result;
}

Point2d operator+(const Point2d& u1, const Point2d& u2)
{
	Point2d result(u1);
	result += u2;
	return result;
}

Point2d operator-(const Point2d& u1, const Point2d& u2)
{
	Point2d result(u1);
	result -= u2;
	return result;
}

bool operator==(const Point2d& u1, const Point2d& u2)
{
	return ( u1.x() == u2.x() and u1.y() == u2.y() );
}

bool operator!=(const Point2d& u1, const Point2d& u2)
{
	return ( u1.x() != u2.x() or u1.y() != u2.y() );
}

double Point2d::Distance(const Point2d& u1, const Point2d& u2)
{
    double x1 = u1.x();
	double y1 = u1.y();

	double x2 = u2.x();
	double y2 = u2.y();

    return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

