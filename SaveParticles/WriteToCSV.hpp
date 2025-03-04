#ifndef WRITETOCSV_HPP
#define WRITETOCSV_HPP
__host__ void writeToCSV(const std::string &filename, const std::vector<std::string> &headers, const double *data, int rows, int cols)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write header row
    for (int i = 0; i < headers.size(); ++i)
    {
        file << headers[i];
        if (i != headers.size() - 1)
        {
            file << ",";
        }
    }
    file << std::endl;

    // Write data
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            file << data[i * cols + j];
            if (j != cols - 1)
            {
                file << ",";
            }
        }
        file << std::endl;
    }

    file.close();
}

#include "PrintInformation.hpp"
#endif