// main_reports.cpp
#include <iostream>
#include <vector>
#include "Point.h"
#include "MeanShift.h"

// Funzione per generare pochi punti (senza cluster predefiniti)
std::vector<Point> generate_few_points() {
    return {
            {0.0, 0.0}, {1.0, 1.0}, {-1.0, -1.0},
            {5.0, 5.0}, {-5.0, -5.0}, {3.0, -3.0},
            {6.0, 5.0}, {-3.0, -2.0}, {1.0, -6.0}
    };
}

int main() {
    std::vector<Point> points = generate_few_points();

    double flat_bandwidth = 2.5;
    double flat_epsilon = 0.1;

    double gaussian_bandwidth = 1.0; // Ridotto per più iterazioni
    double gaussian_epsilon = 0.2;   // Ridotto per maggiore precisione

    // Configurazione per il Kernel FLAT
    MeanShift mean_shift_flat(flat_bandwidth, flat_epsilon, FLAT, false, true, false); // verbose_iterations=true, verbose_points=false
    std::vector<IterationTime> times_flat;
    int cluster_count_flat;
    mean_shift_flat.run(points, "flat_sequential", times_flat, cluster_count_flat);

    // Stampa dei tempi di esecuzione per il Kernel FLAT
    double total_time_flat = 0.0;
    for (const auto& iter_time : times_flat) {
        std::cout << "Iterazione " << iter_time.iteration << ": " << iter_time.iteration_time_ms << " ms\n";
        total_time_flat += iter_time.iteration_time_ms;
    }
    std::cout << "Tempo totale: " << total_time_flat << " ms\n";
    std::cout << "Numero di cluster: " << cluster_count_flat << "\n\n";

    // **Importante**: Ricrea i punti originali per il kernel GAUSSIAN
    points = generate_few_points();

    // Configurazione per il Kernel GAUSSIAN
    MeanShift mean_shift_gaussian(gaussian_bandwidth, gaussian_epsilon, GAUSSIAN, false, true, false); // verbose_iterations=true, verbose_points=false
    std::vector<IterationTime> times_gauss;
    int cluster_count_gauss;
    mean_shift_gaussian.run(points, "gaussian_sequential", times_gauss, cluster_count_gauss);

    // Stampa dei tempi di esecuzione per il Kernel GAUSSIAN
    double total_time_gauss = 0.0;
    for (const auto& iter_time : times_gauss) {
        std::cout << "Iterazione " << iter_time.iteration << ": " << iter_time.iteration_time_ms << " ms\n";
        total_time_gauss += iter_time.iteration_time_ms;
    }
    std::cout << "Tempo totale: " << total_time_gauss << " ms\n";
    std::cout << "Numero di cluster: " << cluster_count_gauss << "\n\n";

    return 0;
}
