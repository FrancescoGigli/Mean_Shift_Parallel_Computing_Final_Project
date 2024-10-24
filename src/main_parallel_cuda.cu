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

// Device Euclidean distance function
__device__ double device_euclidean_distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// CUDA kernel to calculate the mean shift for each point
__global__ void mean_shift_kernel(Point* d_points, Point* d_new_points, int num_points, double bandwidth, double epsilon, char* d_converged, int kernel_type) {
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
void MeanShift::run_cuda(std::vector<Point>& points, const std::string& kernel_name, std::vector<IterationTime>& times, int& cluster_count) {
    size_t num_points = points.size();
    size_t point_size = sizeof(Point) * num_points;

    // Allocate device memory
    Point* d_points;
    Point* d_new_points;
    char* d_converged;
    cudaMalloc(&d_points, point_size);
    cudaMalloc(&d_new_points, point_size);
    cudaMalloc(&d_converged, sizeof(char) * num_points);

    // Copy points from host to device
    cudaMemcpy(d_points, points.data(), point_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

    // Display the number of threads per block
    std::cout << "Number of threads per block: " << threads_per_block << "\n";

    bool all_converged = false;
    int current_iteration = 0;
    const int max_iterations = 100;

    // Save initial iteration (iteration 0)
    save_iteration_to_csv(points, kernel_name, current_iteration, num_points, "cuda");

    while (!all_converged && current_iteration < max_iterations) {
        all_converged = true;
        // Measure iteration start time
        auto iter_start = std::chrono::high_resolution_clock::now();

        // Launch CUDA kernel
        mean_shift_kernel<<<num_blocks, threads_per_block>>>(d_points, d_new_points, num_points, bandwidth, epsilon, d_converged, kernel_type);
        cudaDeviceSynchronize();

        // Copy convergence data back to host
        std::vector<char> converged(num_points);
        cudaMemcpy(converged.data(), d_converged, sizeof(char) * num_points, cudaMemcpyDeviceToHost);

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
        cudaMemcpy(new_points.data(), d_new_points, point_size, cudaMemcpyDeviceToHost);

        if (verbose_iterations) {
            std::cout << "\nIteration " << current_iteration << ":\n";
            std::cout << "Iteration " << current_iteration << " completed in " << iter_duration.count() << " ms.\n";
            std::cout << "Number of converged points: " << num_converged << "/" << num_points << "\n";
            std::cout << "Iteration " << current_iteration << " saved in generated_points/iterations/cuda/" << kernel_name << "/" << num_points << "/mean_shift_result_" << kernel_name << "_cuda_" << num_points << "_iter" << current_iteration << ".csv\n";
        }

        // Save iteration results
        save_iteration_to_csv(new_points, kernel_name, current_iteration, num_points, "cuda");

        // Update points for next iteration
        // Copy updated points back to device
        cudaMemcpy(d_points, d_new_points, point_size, cudaMemcpyDeviceToDevice);
    }

    // Copy final points back to host
    cudaMemcpy(points.data(), d_points, point_size, cudaMemcpyDeviceToHost);

    // Count clusters
    cluster_count = count_clusters(points, bandwidth);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_new_points);
    cudaFree(d_converged);
}

// Main function
int main() {
    std::string data_dir = "generated_points";
    std::vector<size_t> sizes = { 10000,25000,50000,100000 }; // Add more sizes as needed

    for (const auto& size : sizes) {
        std::string filename = data_dir + "/points_" + std::to_string(size) + ".csv";
        std::vector<Point> points = read_points_from_csv(filename);

        if (points.empty()) {
            std::cout << "Error: Unable to read points from " << filename << "\n";
            continue;
        }

        // First, run with FLAT kernel
        {
            // Configuration for FLAT kernel
            double flat_bandwidth = 20.0;
            double flat_epsilon = 1.0;

            // Open output file for this configuration
            std::string output_filename = "cuda_flat_" + std::to_string(size) + "_result.txt";
            std::ofstream output_file(output_filename);
            if (!output_file.is_open()) {
                std::cout << "Error: Unable to open output file " << output_filename << "\n";
                continue;
            }

            // Write results to console and file
            write_output(std::cout, output_file, "Configuration: CUDA, Number of points: " + std::to_string(size) + ", Kernel: FLAT\n");

            MeanShift mean_shift_flat(flat_bandwidth, flat_epsilon, FLAT, true, true, false);
            std::vector<IterationTime> times_flat;
            int cluster_count_flat;
            mean_shift_flat.run_cuda(points, "flat", times_flat, cluster_count_flat);

            // Print execution times and number of clusters
            double total_time_flat = 0.0;
            for (const auto& iter_time : times_flat) {
                std::string iter_message = "Iteration " + std::to_string(iter_time.iteration) + ": " + std::to_string(iter_time.iteration_time_ms) + " ms\n";
                write_output(std::cout, output_file, iter_message);
                total_time_flat += iter_time.iteration_time_ms;
            }
            write_output(std::cout, output_file, "Total time: " + std::to_string(total_time_flat) + " ms\n");
            write_output(std::cout, output_file, "Number of clusters: " + std::to_string(cluster_count_flat) + "\n\n");

            output_file.close();
        }

        // Re-read points for GAUSSIAN kernel
        points = read_points_from_csv(filename);

        // Then, run with GAUSSIAN kernel
        {
            // Configuration for GAUSSIAN kernel
            double gaussian_bandwidth = 1.0;
            double gaussian_epsilon = 0.2;

            // Open output file for this configuration
            std::string output_filename = "cuda_gaussian_" + std::to_string(size) + "_result.txt";
            std::ofstream output_file(output_filename);
            if (!output_file.is_open()) {
                std::cout << "Error: Unable to open output file " << output_filename << "\n";
                continue;
            }

            // Write results to console and file
            write_output(std::cout, output_file, "Configuration: CUDA, Number of points: " + std::to_string(size) + ", Kernel: GAUSSIAN\n");

            MeanShift mean_shift_gaussian(gaussian_bandwidth, gaussian_epsilon, GAUSSIAN, true, true, false);
            std::vector<IterationTime> times_gauss;
            int cluster_count_gauss;
            mean_shift_gaussian.run_cuda(points, "gaussian", times_gauss, cluster_count_gauss);

            // Print execution times and number of clusters
            double total_time_gauss = 0.0;
            for (const auto& iter_time : times_gauss) {
                std::string iter_message = "Iteration " + std::to_string(iter_time.iteration) + ": " + std::to_string(iter_time.iteration_time_ms) + " ms\n";
                write_output(std::cout, output_file, iter_message);
                total_time_gauss += iter_time.iteration_time_ms;
            }
            write_output(std::cout, output_file, "Total time: " + std::to_string(total_time_gauss) + " ms\n");
            write_output(std::cout, output_file, "Number of clusters: " + std::to_string(cluster_count_gauss) + "\n\n");

            output_file.close();
        }
    }

    return 0;
}
