// include/Utils.h
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include "Point.h"

// Funzione per leggere punti da un file CSV
std::vector<Point> read_points_from_csv(const std::string& filename);

// Funzione per aprire un file di output
std::ofstream open_output_file(const std::string& filename);

// Funzione per scrivere output su stream e file
void write_output(std::ostream& console, std::ofstream& file, const std::string& message);

#endif // UTILS_H
