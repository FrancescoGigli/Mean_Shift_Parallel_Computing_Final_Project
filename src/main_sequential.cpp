// main_sequential.cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "Point.h"
#include "MeanShift.h"
#include "Utils.h"

int main() {
    std::vector<size_t> sizes = { 10000, 25000, 50000, 100000 };
    std::string data_dir = "generated_points";

    for (const auto& size : sizes) {
        std::string filename = data_dir + "/points_" + std::to_string(size) + ".csv";
        std::vector<Point> points = read_points_from_csv(filename);
        if (points.empty()) {
            std::cout << "Error: Unable to read points from " << filename << "\n";
            continue;
        }

        // FLAT kernel configuration
        double flat_bandwidth = 20.0;
        double flat_epsilon = 1.0;

        std::string output_filename = "sequential_flat_" + std::to_string(size) + "_result.txt";
        std::ofstream output_file(output_filename);
        if (!output_file.is_open()) {
            std::cout << "Error: Unable to open output file " << output_filename << "\n";
            continue;
        }

        write_output(std::cout, output_file, "Configuration: Sequential, Number of points: " + std::to_string(size) + ", Kernel: FLAT\n");

        MeanShift mean_shift_flat(flat_bandwidth, flat_epsilon, FLAT, false, true, false);
        std::vector<IterationTime> times_flat;
        int cluster_count_flat;
        mean_shift_flat.run(points, "flat", times_flat, cluster_count_flat);

        double total_time_flat = 0.0;
        for (const auto& iter_time : times_flat) {
            std::string iter_message = "Iteration " + std::to_string(iter_time.iteration) + ": " + std::to_string(iter_time.iteration_time_ms) + " ms\n";
            write_output(std::cout, output_file, iter_message);
            total_time_flat += iter_time.iteration_time_ms;
        }
        write_output(std::cout, output_file, "Total time: " + std::to_string(total_time_flat) + " ms\n");
        write_output(std::cout, output_file, "Number of clusters: " + std::to_string(cluster_count_flat) + "\n\n");

        output_file.close();

        // Reload points for GAUSSIAN kernel
        points = read_points_from_csv(filename);

        // GAUSSIAN kernel configuration
        double gaussian_bandwidth = 1.0;
        double gaussian_epsilon = 0.2;

        output_filename = "sequential_gaussian_" + std::to_string(size) + "_result.txt";
        output_file.open(output_filename);
        if (!output_file.is_open()) {
            std::cout << "Error: Unable to open output file " << output_filename << "\n";
            continue;
        }

        write_output(std::cout, output_file, "Configuration: Sequential, Number of points: " + std::to_string(size) + ", Kernel: GAUSSIAN\n");

        MeanShift mean_shift_gaussian(gaussian_bandwidth, gaussian_epsilon, GAUSSIAN, false, true, false);
        std::vector<IterationTime> times_gauss;
        int cluster_count_gauss;
        mean_shift_gaussian.run(points, "gaussian", times_gauss, cluster_count_gauss);

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

    return 0;
}
