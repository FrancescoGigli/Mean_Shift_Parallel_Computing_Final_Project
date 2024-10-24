// Point.h
#ifndef POINT_H
#define POINT_H

// Conditionally include __host__ and __device__ for CUDA compilation
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// Structure to represent a point in 2D space
struct Point {
    double x;
    double y;
    bool converged;

    // Default constructor
    CUDA_HOST_DEVICE Point() : x(0.0), y(0.0), converged(false) {}

    // Parameterized constructor
    CUDA_HOST_DEVICE Point(double x_val, double y_val) : x(x_val), y(y_val), converged(false) {}
};

// Define euclidean_distance as a CUDA_HOST_DEVICE function
CUDA_HOST_DEVICE inline double euclidean_distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

#endif // POINT_H
