#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <vector>
#include <string>
#include "Point.h"

// Enum for kernel types
enum KernelType { FLAT = 0, GAUSSIAN = 1 };

// Struct for iteration time information
struct IterationTime {
    int iteration;
    double iteration_time_ms;
};

// MeanShift class
class MeanShift {
public:
    // Constructor
    MeanShift(double bandwidth, double epsilon, KernelType kernel_type, bool parallel_mode = false,
              bool verbose_iterations = true, bool verbose_points = false, int num_threads = 1, bool use_atomic = false);

    // Function to run the Mean Shift algorithm (Sequential/OpenMP)
    void run(std::vector<Point>& points, const std::string& kernel_name, std::vector<IterationTime>& times, int& cluster_count);

#ifdef USE_CUDA
    // Function to run the Mean Shift algorithm on CUDA
    void run_cuda(std::vector<Point>& points, const std::string& kernel_name,
                  std::vector<IterationTime>& times, int& cluster_count, int threads_per_block);
#endif

private:
    double bandwidth;
    double epsilon;
    KernelType kernel_type;
    bool parallel;
    bool verbose_iterations;
    bool verbose_points;
    int num_threads;
    bool use_atomic;

    // Kernel function based on selected kernel type
    double kernel_function(double distance);

    // Compute the mean shift for a given point
    Point compute_mean_shift(Point& point, const std::vector<Point>& points);

    // Count the number of unique clusters
    int count_clusters(const std::vector<Point>& points, double cluster_epsilon);

    // Save iteration data to a CSV file
    void save_iteration_to_csv(const std::vector<Point>& points, const std::string& kernel_name,
                               int iteration, size_t dataset_size, const std::string& mode);
};

#endif // MEANSHIFT_H
