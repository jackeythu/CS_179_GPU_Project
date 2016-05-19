#ifndef Point2d_H
#define Point2d_H

#include <cmath>

//Point2d.h
class Point {
public:
    Point() {};
	virtual ~Point() {};
	virtual double norm2() const=0;
};
    
class Point2d : public Point {
private:
    double x_ = 0.0;
    double y_ = 0.0;

public:
    //constructor, destructor and assignment operator
    Point2d() = default; 
    Point2d(double x, double y);

	~Point2d() = default;
    Point2d(const Point2d&) = default;
	Point2d& operator=(const Point2d&) = default;
    
    //scalar multiplication and division
    Point2d& operator*=(double);
	Point2d& operator/=(double);
        
    //vector addition and subtraction
	Point2d& operator+=(const Point2d&);
	Point2d& operator-=(const Point2d&);
	

	//subscript: index 0 and 1 returns reference to x_ and y_
	const double& operator[](std::size_t) const;
	double& operator[](std::size_t);

    //interface to x_, y_
    const double x() const { return x_; }
    const double y() const { return y_; }

	//L2 nom and its square
	double norm() const { return sqrt(x_*x_ + y_*y_); }
	double norm2() const { return (x_*x_ + y_*y_); }

	//Euclidean Distance
    static double Distance(const Point2d& u1, const Point2d& u2);
};

Point2d operator*(const Point2d& u, double);
Point2d operator*(double, const Point2d& u);
double operator*(const Point2d& u1, const Point2d& u2);
Point2d operator/(const Point2d& u, double);
Point2d operator+(const Point2d& u1, const Point2d& u2);
Point2d operator-(const Point2d& u1, const Point2d& u2);
bool operator==(const Point2d& u1, const Point2d& u2);
bool operator!=(const Point2d& u1, const Point2d& u2);

#endif
