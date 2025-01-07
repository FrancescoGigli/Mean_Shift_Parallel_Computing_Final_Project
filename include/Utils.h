// include/Utils.h
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include "Point.h"


std::vector<Point> read_points_from_csv(const std::string& filename);

void write_output(std::ostream& console, std::ofstream& file, const std::string& message);

#endif // UTILS_H
