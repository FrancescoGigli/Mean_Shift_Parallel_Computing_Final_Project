// main_parallel_openMP.cpp

#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "Point.h"
#include "MeanShift.h"
#include "Utils.h"

int main() {

    //INITIALIZATION
    std::vector<size_t> sizes = {10000, 25000, 50000, 100000};
    std::string data_dir = "generated_points";
    int num_threads = 4;

    omp_set_dynamic(0);        // Disables dynamic adjustment of threads
    omp_set_num_threads(num_threads); // Sets a fixed number of threads

    std::vector<std::string> sync_types = {"critical", "atomic"};

    // MEAN-SHIFT over Dataset Sizes (10000, 25000, 50000, 100000), Kernel Types (Flat, Gaussian), Synchronization Methods (Atomic, Critical)
    for (const auto& size : sizes) {
        std::string filename;
        filename.append(data_dir).append("/points_").append(std::to_string(size)).append(".csv");

        std::vector<Point> original_points = read_points_from_csv(filename);

        // Check if original points are empty
        if (original_points.empty()) {
            std::cout << "Error: Unable to read points from " << filename << "\n";
            continue;
        }

        std::vector<std::pair<std::string, KernelType>> kernels = {
                {"flat", FLAT},
                {"gaussian", GAUSSIAN}
        };

        for (const auto& kernel_pair : kernels) {
            std::string kernel_name = kernel_pair.first;
            KernelType kernel_type = kernel_pair.second;

            for (const auto& sync_type : sync_types) {
                std::vector<Point> points = original_points;

                double bandwidth = (kernel_type == FLAT) ? 20.0 : 1.0;
                double epsilon = (kernel_type == FLAT) ? 1.0 : 0.2;

                std::string output_filename;
                output_filename.append("parallel_").append(kernel_name).append("_").append(sync_type).append("_").append(std::to_string(size)).append("_result.txt");

                std::ofstream output_file(output_filename);
                if (!output_file.is_open()) {
                    std::cout << "Error: Unable to open output file " << output_filename << "\n";
                    continue;
                }

                int used_threads = omp_get_max_threads(); //number of used thread


                write_output(std::cout, output_file, "Configuration: Parallel, Number of points: " + std::to_string(size) + ", Kernel: " + kernel_name + ", Sync: " + sync_type + "\n");
                write_output(std::cout, output_file, "Number of threads used: " + std::to_string(used_threads) + "\n");

                bool use_atomic = (sync_type == "atomic");


                MeanShift mean_shift(bandwidth, epsilon, kernel_type, true, true, false, num_threads, use_atomic);

                std::vector<IterationTime> times;
                int cluster_count;

                mean_shift.run(points, kernel_name, times, cluster_count);

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


    //Dataset-specific test with varying threads from 2 to 16 on 100000 points
    size_t large_dataset_size = 100000;
    std::string large_filename;
    large_filename.append(data_dir).append("/points_").append(std::to_string(large_dataset_size)).append(".csv");

    std::vector<Point> large_points = read_points_from_csv(large_filename);
    if (large_points.empty()) {
        std::cout << "Error: Unable to read points from " << large_filename << "\n";
        return 1;
    }

    std::vector<std::pair<std::string, KernelType>> kernels_large = {
            {"flat", FLAT},
            {"gaussian", GAUSSIAN}
    };

    std::vector<std::string> sync_types_large = {"critical", "atomic"};

    for (const auto& kernel_pair : kernels_large) {
        std::string kernel_name = kernel_pair.first;
        KernelType kernel_type = kernel_pair.second;

        for (const auto& sync_type : sync_types_large) {
            for (int threads : {2, 4, 6, 8, 10, 12, 16}) {
                std::vector<Point> points = large_points;

                omp_set_num_threads(threads);

                double bandwidth = (kernel_type == FLAT) ? 20.0 : 1.0;
                double epsilon = (kernel_type == FLAT) ? 1.0 : 0.2;

                std::string output_filename;
                output_filename.append("parallel_").append(kernel_name).append("_").append(sync_type).append("_100000_threads_").append(std::to_string(threads)).append("_result.txt");

                std::ofstream output_file(output_filename);
                if (!output_file.is_open()) {
                    std::cout << "Error: Unable to open output file " << output_filename << "\n";
                    continue;
                }

                write_output(std::cout, output_file, "Configuration: Parallel, Number of points: " + std::to_string(large_dataset_size) + ", Kernel: " + kernel_name + ", Sync: " + sync_type + ", Threads: " + std::to_string(threads) + "\n");

                bool use_atomic = (sync_type == "atomic");

                MeanShift mean_shift(bandwidth, epsilon, kernel_type, true, true, false, threads, use_atomic);
                std::vector<IterationTime> times;
                int cluster_count;
                mean_shift.run(points, kernel_name, times, cluster_count);

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
