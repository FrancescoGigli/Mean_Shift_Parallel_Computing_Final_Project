// src/Utils.cpp
#include "Utils.h"
#include <sstream>
#include <iostream>

// Function to read points from a CSV file
std::vector<Point> read_points_from_csv(const std::string& filename) {
    std::vector<Point> points;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        return points;
    }

    std::string line;
    // Ignore the header if present
    if (std::getline(file, line)) {
        if (line.find("x,y") == std::string::npos && line.find("bandwidth,iteration,point_id,x,y") == std::string::npos) {
            // If the first line is not a header, process it
            std::stringstream ss(line);
            std::string x_str, y_str;
            if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ',')) {
                points.emplace_back(std::stod(x_str), std::stod(y_str));
            }
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ',')) {
            try {
                double x = std::stod(x_str);
                double y = std::stod(y_str);
                points.emplace_back(x, y);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid line in CSV: " << line << "\n";
            }
        }
    }

    file.close();
    return points;
}


// Function to write output to both console and file
void write_output(std::ostream& console, std::ofstream& file, const std::string& message) {
    console << message;
    if (file.is_open()) {
        file << message;
    }
}
