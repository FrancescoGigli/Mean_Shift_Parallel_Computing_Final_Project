// main_parallel_openMP.cpp

#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "Point.h"
#include "MeanShift.h"
#include "Utils.h"

int main() {
    std::vector<size_t> sizes = {10000, 25000, 50000, 100000};
    std::string data_dir = "generated_points";
    int num_threads = 4; // Set the number of threads for standard runs

    // Disable dynamic adjustment of threads
    omp_set_dynamic(0);
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Synchronization types
    std::vector<std::string> sync_types = {"critical", "atomic"};

    for (const auto& size : sizes) {
        std::string filename = data_dir + "/points_" + std::to_string(size) + ".csv";

        // Load points
        std::vector<Point> original_points = read_points_from_csv(filename);
        if (original_points.empty()) {
            std::cout << "Error: Unable to read points from " << filename << "\n";
            continue;
        }

        // Kernel configurations
        std::vector<std::pair<std::string, KernelType>> kernels = {
                {"flat", FLAT},
                {"gaussian", GAUSSIAN}};

        for (const auto& kernel_pair : kernels) {
            std::string kernel_name = kernel_pair.first;
            KernelType kernel_type = kernel_pair.second;

            // Synchronization types loop
            for (const auto& sync_type : sync_types) {
                // Clone the original points for each run
                std::vector<Point> points = original_points;

                // Set bandwidth and epsilon based on kernel
                double bandwidth = (kernel_type == FLAT) ? 20.0 : 1.0;
                double epsilon = (kernel_type == FLAT) ? 1.0 : 0.2;

                // Open the output file for this configuration
                std::string output_filename = "parallel_" + kernel_name + "_" + sync_type + "_" + std::to_string(size) + "_result.txt";
                std::ofstream output_file(output_filename);
                if (!output_file.is_open()) {
                    std::cout << "Error: Unable to open output file " << output_filename << "\n";
                    continue;
                }

                // Write results to both console and file
                int used_threads = omp_get_max_threads();  // Get the number of threads OpenMP will use
                write_output(std::cout, output_file, "Configuration: Parallel, Number of points: " + std::to_string(size) + ", Kernel: " + kernel_name + ", Sync: " + sync_type + "\n");
                write_output(std::cout, output_file, "Number of threads used: " + std::to_string(used_threads) + "\n");

                // Determine whether to use atomic or not
                bool use_atomic = (sync_type == "atomic");

                MeanShift mean_shift(bandwidth, epsilon, kernel_type, true, true, false, num_threads, use_atomic);
                std::vector<IterationTime> times;
                int cluster_count;
                mean_shift.run(points, kernel_name, times, cluster_count);

                // Print execution times and cluster count
                double total_time = 0.0;
                for (const auto& iter_time : times) {
                    std::string iter_message = "Iteration " + std::to_string(iter_time.iteration) + ": " + std::to_string(iter_time.iteration_time_ms) + " ms\n";
                    write_output(std::cout, output_file, iter_message);
                    total_time += iter_time.iteration_time_ms;
                }
                write_output(std::cout, output_file, "Total time: " + std::to_string(total_time) + " ms\n");
                write_output(std::cout, output_file, "Number of clusters: " + std::to_string(cluster_count) + "\n\n");

                output_file.close();
            }
        }
    }

    // Additional run for 100000-point dataset with varying thread counts
    size_t large_dataset_size = 100000;
    std::string large_filename = data_dir + "/points_" + std::to_string(large_dataset_size) + ".csv";

    // Load points for the large dataset
    std::vector<Point> large_points = read_points_from_csv(large_filename);
    if (large_points.empty()) {
        std::cout << "Error: Unable to read points from " << large_filename << "\n";
        return 1;
    }

    // Kernel configuration (can be extended for both flat and Gaussian kernels)
    std::vector<std::pair<std::string, KernelType>> kernels_large = {
            {"flat", FLAT},
            {"gaussian", GAUSSIAN}};

    // Synchronization types
    std::vector<std::string> sync_types_large = {"critical", "atomic"};

    for (const auto& kernel_pair : kernels_large) {
        std::string kernel_name = kernel_pair.first;
        KernelType kernel_type = kernel_pair.second;

        for (const auto& sync_type : sync_types_large) {
            // Loop over the specific number of threads {2, 4, 6, 8, 10, 12, 16}
            for (int threads : {2, 4, 6, 8, 10, 12, 16}) {
                std::vector<Point> points = large_points;

                // Set OpenMP threads
                omp_set_num_threads(threads);

                // Set bandwidth and epsilon based on kernel
                double bandwidth = (kernel_type == FLAT) ? 20.0 : 1.0;
                double epsilon = (kernel_type == FLAT) ? 1.0 : 0.2;

                // Open the output file for the large dataset with varying threads
                std::string output_filename = "parallel_" + kernel_name + "_" + sync_type + "_100000_threads_" + std::to_string(threads) + "_result.txt";
                std::ofstream output_file(output_filename);
                if (!output_file.is_open()) {
                    std::cout << "Error: Unable to open output file " << output_filename << "\n";
                    continue;
                }

                // Write results to both console and file
                write_output(std::cout, output_file, "Configuration: Parallel, Number of points: " + std::to_string(large_dataset_size) + ", Kernel: " + kernel_name + ", Sync: " + sync_type + ", Threads: " + std::to_string(threads) + "\n");

                // Determine whether to use atomic or not
                bool use_atomic = (sync_type == "atomic");

                MeanShift mean_shift(bandwidth, epsilon, kernel_type, true, true, false, threads, use_atomic);
                std::vector<IterationTime> times;
                int cluster_count;
                mean_shift.run(points, kernel_name, times, cluster_count);

                // Print execution times and cluster count
                double total_time = 0.0;
                for (const auto& iter_time : times) {
                    std::string iter_message = "Iteration " + std::to_string(iter_time.iteration) + ": " + std::to_string(iter_time.iteration_time_ms) + " ms\n";
                    write_output(std::cout, output_file, iter_message);
                    total_time += iter_time.iteration_time_ms;
                }
                write_output(std::cout, output_file, "Total time: " + std::to_string(total_time) + " ms\n");
                write_output(std::cout, output_file, "Number of clusters: " + std::to_string(cluster_count) + "\n\n");

                output_file.close();
            }
        }
    }

    return 0;
}
