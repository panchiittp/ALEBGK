#include <iostream>
#include <cmath>
#include <algorithm>  // For std::min

struct BGKParticle {
    double x, y, ux, uy;
    int voxel;
};

int computeVoxelIndex(double x, double y, double voxel_size, int total_voxels_x) {
    int voxel_x = std::min(static_cast<int>(x / voxel_size), total_voxels_x - 1);
    int voxel_y = std::min(static_cast<int>(y / voxel_size), total_voxels_x - 1);
    return voxel_x + total_voxels_x * voxel_y;
}

int main() {
    double spacing = 0.02;    // Particle spacing
    double voxel_size = 0.1;  // Voxel size
    int num_x = static_cast<int>(1.0 / spacing) + 1;
    int num_y = static_cast<int>(1.0 / spacing) + 1;
    int total_particles = num_x * num_y;
    int total_voxels_x = static_cast<int>(1.0 / voxel_size) + 1;
    int total_voxels = total_voxels_x * total_voxels_x;

    BGKParticle* particles = new BGKParticle[total_particles];
    int* voxel_counts = new int[total_voxels](); // Initialize to zero
    
    int particle_index = 0;

    // Generate particles
    for (int i = 0; i < num_x; ++i) {
        for (int j = 0; j < num_y; ++j) {
            double x = i * spacing;
            double y = j * spacing;
            if (x >= 1.0) x = 1.0 - 1e-6; // Ensure it falls inside [0,1]
            if (y >= 1.0) y = 1.0 - 1e-6;

            particles[particle_index] = {x, y, 0.0, 0.0, 0};
            
            // Compute voxel index
            particles[particle_index].voxel = computeVoxelIndex(x, y, voxel_size, total_voxels_x);
            
            // Increment voxel count
            voxel_counts[particles[particle_index].voxel]++;
            
            ++particle_index;
        }
    }

    // Print the number of particles per voxel
    for (int i = 0; i < total_voxels; ++i) {
        std::cout << "Voxel " << i << " has " << voxel_counts[i] << " particles." << std::endl;
    }

    delete[] particles;
    delete[] voxel_counts;
    return 0;
}
