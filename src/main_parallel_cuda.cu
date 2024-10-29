// main_parallel_cuda.cu

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// Include custom headers
#include "MeanShift.h"
#include "Point.h"
#include "Utils.h"

// Define the CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device Euclidean distance function
__device__ double device_euclidean_distance(const Point a, const Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// CUDA kernel to calculate the mean shift for each point
__global__ void mean_shift_kernel(Point* d_points, Point* d_new_points, int num_points, double bandwidth,
                                  double epsilon, char* d_converged, int kernel_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point point = d_points[idx];
    double sum_x = 0.0, sum_y = 0.0;
    double sum_weight = 0.0;

    // Calculate the weighted sum of neighboring points
    for (int i = 0; i < num_points; ++i) {
        double dist = device_euclidean_distance(point, d_points[i]);
        double weight = 0.0;
        if (kernel_type == FLAT) { // FLAT kernel
            weight = (dist < bandwidth) ? 1.0 : 0.0;
        } else { // GAUSSIAN kernel
            weight = exp(- (dist * dist) / (2 * bandwidth * bandwidth));
        }

        if (weight > 0) {
            sum_x += d_points[i].x * weight;
            sum_y += d_points[i].y * weight;
            sum_weight += weight;
        }
    }

    // Update the coordinates of the new point
    if (sum_weight > 0) {
        d_new_points[idx].x = sum_x / sum_weight;
        d_new_points[idx].y = sum_y / sum_weight;

        // Calculate the shift distance
        double shift_distance = device_euclidean_distance(point, d_new_points[idx]);
        d_converged[idx] = (shift_distance < epsilon);
    } else {
        d_new_points[idx] = point;
        d_converged[idx] = 1;  // Consider it converged if no movement
    }
}

// MeanShift class implementation (only run_cuda method)
void MeanShift::run_cuda(std::vector<Point>& points, const std::string& kernel_name,
                         std::vector<IterationTime>& times, int& cluster_count, int threads_per_block) {
    size_t num_points = points.size();
    size_t point_size = sizeof(Point) * num_points;

    // Allocate device memory with error checking
    Point* d_points;
    Point* d_new_points;
    char* d_converged;
    CUDA_CHECK(cudaMalloc(&d_points, point_size));
    CUDA_CHECK(cudaMalloc(&d_new_points, point_size));
    CUDA_CHECK(cudaMalloc(&d_converged, sizeof(char) * num_points));

    // Copy points from host to device with error checking
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), point_size, cudaMemcpyHostToDevice));

    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

    // Display the number of threads per block
    std::cout << "Number of threads per block: " << threads_per_block << "\n";

    bool all_converged = false;
    int current_iteration = 0;
    const int max_iterations = 100;

    // Construct mode string with threads_per_block
    std::string mode = "cuda_threads_" + std::to_string(threads_per_block);

    // Save initial iteration (iteration 0)
    save_iteration_to_csv(points, kernel_name, current_iteration, num_points, mode);

    while (!all_converged && current_iteration < max_iterations) {
        all_converged = true;

        // Measure iteration start time
        auto iter_start = std::chrono::high_resolution_clock::now();

        // Launch CUDA kernel with error checking
        mean_shift_kernel<<<num_blocks, threads_per_block>>>(d_points, d_new_points, num_points,
                                                             bandwidth, epsilon, d_converged, kernel_type);
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
        CUDA_CHECK(cudaDeviceSynchronize()); // Synchronize and check for errors

        // Copy convergence data back to host
        std::vector<char> converged(num_points);
        CUDA_CHECK(cudaMemcpy(converged.data(), d_converged, sizeof(char) * num_points, cudaMemcpyDeviceToHost));

        // Check convergence and count number of converged points
        int num_converged = 0;
        for (size_t i = 0; i < num_points; ++i) {
            if (converged[i]) {
                num_converged++;
            } else {
                all_converged = false;
            }
        }

        // Measure iteration end time
        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_duration = iter_end - iter_start;

        current_iteration++;
        times.push_back(IterationTime{ current_iteration, iter_duration.count() });

        // Copy new points back to host for saving
        std::vector<Point> new_points(num_points);
        CUDA_CHECK(cudaMemcpy(new_points.data(), d_new_points, point_size, cudaMemcpyDeviceToHost));

        if (verbose_iterations) {
            std::cout << "\nIteration " << current_iteration << ":\n";
            std::cout << "Iteration " << current_iteration << " completed in " << iter_duration.count() << " ms.\n";
            std::cout << "Number of converged points: " << num_converged << "/" << num_points << "\n";
            std::cout << "Iteration " << current_iteration << " saved in generated_points/iterations/cuda/"
                      << kernel_name << "/" << num_points << "/" << mode << "/mean_shift_result_"
                      << kernel_name << "_cuda_" << num_points << "_threads_" << threads_per_block << "_iter"
                      << current_iteration << ".csv\n";
        }

        // Save iteration results
        save_iteration_to_csv(new_points, kernel_name, current_iteration, num_points, mode);

        // Update points for next iteration
        CUDA_CHECK(cudaMemcpy(d_points, d_new_points, point_size, cudaMemcpyDeviceToDevice));
    }

    // Copy final points back to host
    CUDA_CHECK(cudaMemcpy(points.data(), d_points, point_size, cudaMemcpyDeviceToHost));

    // Count clusters
    cluster_count = count_clusters(points, bandwidth);

    // Free device memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_new_points));
    CUDA_CHECK(cudaFree(d_converged));
}

// Main function
int main() {
    std::string data_dir = "generated_points";
    std::vector<size_t> sizes = { 10000, 25000, 50000, 100000 }; // Dataset sizes
    std::vector<int> thread_options = { 64, 128, 256, 512, 1024 }; // Thread configurations

    for (const auto& size : sizes) {
        std::string filename = data_dir + "/points_" + std::to_string(size) + ".csv";
        std::vector<Point> points_original = read_points_from_csv(filename);

        if (points_original.empty()) {
            std::cout << "Error: Unable to read points from " << filename << "\n";
            continue;
        }

        // Run for both FLAT and GAUSSIAN kernels
        for (const auto& kernel_type : { FLAT, GAUSSIAN }) {
            std::string kernel_name = (kernel_type == FLAT) ? "flat" : "gaussian";

            // Configuration for kernels
            double bandwidth = (kernel_type == FLAT) ? 20.0 : 1.0;
            double epsilon = (kernel_type == FLAT) ? 1.0 : 0.2;

            for (int threads_per_block : thread_options) {
                // Create a copy of the original points
                std::vector<Point> points = points_original;

                std::cout << "Running with " << threads_per_block << " threads per block\n";

                // Output file name
                std::string output_filename = "cuda_" + kernel_name + "_" + std::to_string(size) +
                                              "_threads_" + std::to_string(threads_per_block) + "_result.txt";
                std::ofstream output_file(output_filename);
                if (!output_file.is_open()) {
                    std::cout << "Error: Unable to open output file " << output_filename << "\n";
                    continue;
                }

                // Construct mode string with threads_per_block
                std::string mode = "cuda_threads_" + std::to_string(threads_per_block);

                // Write configuration to console and file
                write_output(std::cout, output_file, "Configuration: CUDA, Number of points: " + std::to_string(size) +
                                                     ", Kernel: " + kernel_name + ", Threads per block: " + std::to_string(threads_per_block) + "\n");

                // Initialize MeanShift object
                MeanShift mean_shift(bandwidth, epsilon, kernel_type, true, true, false);

                std::vector<IterationTime> times;
                int cluster_count;

                // Run CUDA Mean Shift
                mean_shift.run_cuda(points, kernel_name, times, cluster_count, threads_per_block);

                // Accumulate and write iteration times
                double total_time = 0.0;
                for (const auto& iter_time : times) {
                    std::string iter_message = "Iteration " + std::to_string(iter_time.iteration) + ": " +
                                               std::to_string(iter_time.iteration_time_ms) + " ms\n";
                    write_output(std::cout, output_file, iter_message);
                    total_time += iter_time.iteration_time_ms;
                }

                // Write total time and number of clusters
                write_output(std::cout, output_file, "Total time: " + std::to_string(total_time) + " ms\n");
                write_output(std::cout, output_file, "Number of clusters: " + std::to_string(cluster_count) + "\n\n");

                output_file.close();
            }
        }
    }

    return 0;
}
