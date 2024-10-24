#include "MeanShift.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <set>
#include <chrono>
#include <omp.h>
#include <filesystem>

namespace fs = std::filesystem;

// Constructor
MeanShift::MeanShift(double bandwidth, double epsilon, KernelType kernel_type, bool parallel_mode,
                     bool verbose_iterations, bool verbose_points, int num_threads, bool use_atomic)
        : bandwidth(bandwidth), epsilon(epsilon), kernel_type(kernel_type), parallel(parallel_mode),
          verbose_iterations(verbose_iterations), verbose_points(verbose_points), num_threads(num_threads), use_atomic(use_atomic) {}

// Kernel function implementation
double MeanShift::kernel_function(double distance) {
    if (kernel_type == FLAT) {
        return (distance < bandwidth) ? 1.0 : 0.0;
    } else { // GAUSSIAN
        return std::exp(-distance * distance / (2 * bandwidth * bandwidth));
    }
}

// Compute the mean shift for a given point
Point MeanShift::compute_mean_shift(Point& point, const std::vector<Point>& points) {
    double sum_x = 0.0, sum_y = 0.0;
    double sum_weight = 0.0;

    if (use_atomic) {
#pragma omp parallel for if(parallel) num_threads(num_threads) reduction(+:sum_x, sum_y, sum_weight)
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            double dist = euclidean_distance(point, points[i]);
            double weight = kernel_function(dist);
            if (weight > 0) {
                sum_x += points[i].x * weight;
                sum_y += points[i].y * weight;
                sum_weight += weight;
            }
        }
    } else {
#pragma omp parallel for if(parallel) num_threads(num_threads)
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            double dist = euclidean_distance(point, points[i]);
            double weight = kernel_function(dist);
            if (weight > 0) {
#pragma omp critical
                {
                    sum_x += points[i].x * weight;
                    sum_y += points[i].y * weight;
                    sum_weight += weight;
                }
            }
        }
    }

    if (sum_weight == 0.0) return point;

    return Point(sum_x / sum_weight, sum_y / sum_weight);
}

// Count the number of unique clusters
int MeanShift::count_clusters(const std::vector<Point>& points, double cluster_epsilon) {
    std::set<std::pair<int, int>> unique_clusters;

    for (const auto& point : points) {
        int cluster_x = static_cast<int>(std::round(point.x / cluster_epsilon));
        int cluster_y = static_cast<int>(std::round(point.y / cluster_epsilon));
        unique_clusters.emplace(cluster_x, cluster_y);
    }

    return static_cast<int>(unique_clusters.size());
}

// Save iteration data to a CSV file
void MeanShift::save_iteration_to_csv(const std::vector<Point>& points, const std::string& kernel_name, int iteration, size_t dataset_size, const std::string& mode) {
    std::string iter_dir = "generated_points/iterations/" + mode + "/" + kernel_name + "/" + std::to_string(dataset_size);
    fs::create_directories(iter_dir);

    std::string filename = iter_dir + "/mean_shift_result_" + kernel_name + "_" + mode + "_" + std::to_string(dataset_size) + "_iter" + std::to_string(iteration) + ".csv";

    std::ofstream file(filename);
    if (file.is_open()) {
        file << "bandwidth,iteration,point_id,x,y\n";
        for (size_t j = 0; j < points.size(); ++j) {
            file << bandwidth << "," << iteration << "," << j << "," << points[j].x << "," << points[j].y << "\n";
        }
        file.close();
        if (verbose_iterations) {
            std::cout << "Iteration " << iteration << " saved in " << filename << "\n";
        }
    } else {
        std::cerr << "Error opening file: " << filename << "\n";
    }
}

// Runs the MeanShift algorithm (Sequential/OpenMP)
void MeanShift::run(std::vector<Point>& points, const std::string& kernel_name, std::vector<IterationTime>& times, int& cluster_count) {
    size_t dataset_size = points.size();
    std::string mode = parallel ? "parallel" : "sequential";

    // Save the initial iteration (iteration 0)
    save_iteration_to_csv(points, kernel_name, 0, dataset_size, mode);

    bool all_converged = false;
    int current_iteration = 0;
    const int max_iterations = 100;

    // Define cluster_epsilon (can be adjusted as needed)
    double cluster_epsilon = bandwidth; // Example: equal to bandwidth

    while (!all_converged && current_iteration < max_iterations) {
        all_converged = true;
        std::vector<Point> new_points = points;

        if (verbose_iterations) {
            std::cout << "Iteration " << current_iteration + 1 << ":\n";
        }

        auto iter_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic) if(parallel) num_threads(num_threads)
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            if (points[i].converged) continue;

            Point new_point = compute_mean_shift(points[i], points);
            double shift_distance = euclidean_distance(points[i], new_point);

            if (shift_distance >= epsilon) {
#pragma omp critical
                {
                    all_converged = false;
                }
            } else {
                new_point.converged = true;
            }

            new_points[i] = new_point;

            if (verbose_points) {
#pragma omp critical
                {
                    int thread_id = omp_get_thread_num();
                    std::cout << "  Thread " << thread_id << ": Point " << i << ": (" << points[i].x << ", " << points[i].y << ") -> ("
                              << new_points[i].x << ", " << new_points[i].y << "), shift distance: " << shift_distance << "\n";
                }
            }
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_duration = iter_end - iter_start;

        times.push_back(IterationTime{current_iteration + 1, iter_duration.count()});

        points = new_points;
        current_iteration++;

        if (verbose_iterations) {
            std::cout << "Iteration " << current_iteration << " completed in " << iter_duration.count() << " ms.\n";
            int converged_points = 0;
            for (const auto& point : points) {
                if (point.converged) converged_points++;
            }
            std::cout << "Number of converged points: " << converged_points << "/" << points.size() << "\n\n";
        }

        // Save the current iteration to a CSV
        save_iteration_to_csv(points, kernel_name, current_iteration, dataset_size, mode);
    }

    cluster_count = count_clusters(points, cluster_epsilon);
    if (verbose_iterations) {
        int thread_id = omp_get_thread_num();
        std::cout << "Thread " << thread_id << ": Convergence reached in " << current_iteration << " iterations (Max: " << max_iterations << ").\n";
        std::cout << "Thread " << thread_id << ": Number of clusters formed: " << cluster_count << "\n";
    }
}
