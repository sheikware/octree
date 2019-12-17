#include <iostream>
#include <cstdlib>
#include <time.h>

#include "Octree.hpp"
#include "utils.h"

/** Example 1: Searching radius neighbors with default access by public x,y,z variables.
 *
 * \author behley
 */

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "filename of point cloud missing." << std::endl;
    return -1;
  }
  std::string filename = argv[1];

  Eigen::Matrix3Xd points;
  readPoints(filename, points);
  std::cout << "Read " << points.cols() << " points." << std::endl;
  if (points.cols() == 0)
  {
    std::cerr << "Empty point cloud." << std::endl;
    return -1;
  }

  int64_t begin, end;

  // initializing the Octree with points from point cloud.
  unibn::Octree octree;
  unibn::OctreeParams params;
  octree.initialize(points);

  // radiusNeighbors returns indexes to neighboring points.
  std::vector<size_t> results;
  const Eigen::Vector3d& q = points.col(0);
  octree.radiusNeighbors<unibn::L2Distance>(q, 0.2f, results);
  std::cout << results.size() << " radius neighbors (r = 0.2m) found for (" << q[0] << ", " << q[1] << "," << q[2] << ")"
            << std::endl;
  for (size_t i = 0; i < results.size(); ++i)
  {
    const Eigen::Vector3d& p = points.col(results[i]);
    std::cout << "  " << results[i] << ": (" << p[0] << ", " << p[1] << ", " << p[2] << ") => "
              << std::sqrt(unibn::L2Distance::compute(p, q)) << std::endl;
  }

  // performing queries for each point in point cloud
  begin = clock();
  for (size_t i = 0; i < points.cols(); ++i)
  {
    octree.radiusNeighbors<unibn::L2Distance>(points.col(i), 0.5f, results);
  }
  end = clock();
  double search_time = ((double)(end - begin) / CLOCKS_PER_SEC);
  std::cout << "Searching for all radius neighbors (r = 0.5m) took " << search_time << " seconds." << std::endl;

  octree.clear();

  return 0;
}
