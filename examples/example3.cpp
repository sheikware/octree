#include <iostream>
#include <cstdlib>
#include <time.h>

#include "Octree.hpp"
#include "utils.h"

/** Example for a templated descriptor computation
 * \author behley
 */

class SimpleDescriptor
{
 public:
  SimpleDescriptor(double r, size_t dim) : radius_(r), dim_(dim)
  {
  }

  void compute(const Eigen::Vector3d& query, const Eigen::Matrix3Xd& pts, const unibn::Octree& oct,
               std::vector<double>& descriptor)
  {
    memset(&descriptor[0], 0, dim_);

    std::vector<size_t> neighbors;
    std::vector<double> distances;

    // template is needed to tell the compiler that radiusNeighbors is a method.
    oct.template radiusNeighbors<unibn::MaxDistance>(query, radius_, neighbors, distances);
    for (size_t i = 0; i < neighbors.size(); ++i) descriptor[distances[i] / radius_ * dim_] += 1;
  }

  uint32_t dim() const
  {
    return dim_;
  }

 protected:
  double radius_;
  size_t dim_;
};

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "filename of point cloud missing." << std::endl;
    return -1;
  }
  std::string filename = argv[1];

  std::shared_ptr<Eigen::Matrix3Xd> points(new Eigen::Matrix3Xd);
  readPoints(filename, *points);
  std::cout << "Read " << points->cols() << " points." << std::endl;
  if (points->cols() == 0)
  {
    std::cerr << "Empty point cloud." << std::endl;
    return -1;
  }

  int64_t begin, end;

  // initializing the Octree with points from point cloud.
  unibn::Octree octree;
  unibn::OctreeParams params;
  octree.initialize(points);

  SimpleDescriptor desc(0.5, 5);
  std::vector<double> values(desc.dim());
  // performing descriptor computations for each point in point cloud
  begin = clock();
  for (size_t i = 0; i < points->cols(); ++i)
  {
    desc.compute(points->col(i), *points, octree, values);
  }
  end = clock();
  double search_time = ((double)(end - begin) / CLOCKS_PER_SEC);
  std::cout << "Computing simple descriptor for all points took " << search_time << " seconds." << std::endl;

  octree.clear();

  return 0;
}
