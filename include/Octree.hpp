#ifndef UNIBN_OCTREE_H_
#define UNIBN_OCTREE_H_

// Copyright (c) 2015 Jens Behley, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <stdint.h>
#include <cassert>
#include <cmath>
#include <cstring>  // memset.
#include <limits>
#include <vector>
#include <numeric>
#include <Eigen/Core>
#include <memory>

// needed for gtest access to protected/private members ...
namespace
{
class OctreeTest;
}

namespace unibn
{

/**
 * Some generic distances: Manhattan, (squared) Euclidean, and Maximum distance.
 *
 * A Distance has to implement the methods
 * 1. compute of two points p and q to compute and return the distance between two points, and
 * 2. norm of x,y,z coordinates to compute and return the norm of a point p = (x,y,z)
 * 3. sqr and sqrt of value to compute the correct radius if a comparison is performed using squared norms (see
 *L2Distance)...
 */
struct L1Distance
{
  static inline double compute(const Eigen::Vector3d& p, const Eigen::Vector3d& q)
  {
    Eigen::Vector3d d = p - q;
    return std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]);
  }

  static inline double norm(const Eigen::Vector3d& p)
  {
    return std::abs(p[0]) + std::abs(p[1]) + std::abs(p[2]);
  }

  static inline double sqr(double r)
  {
    return r;
  }

  static inline double sqrt(double r)
  {
    return r;
  }
};

struct L2Distance
{
  static inline double compute(const Eigen::Vector3d& p, const Eigen::Vector3d& q)
  {
    Eigen::Vector3d d = p - q;
    return d.squaredNorm();
  }

  static inline double norm(const Eigen::Vector3d& p)
  {
    return p.squaredNorm();
  }

  static inline double sqr(double r)
  {
    return r * r;
  }

  static inline double sqrt(double r)
  {
    return std::sqrt(r);
  }
};

struct MaxDistance
{
  static inline double compute(const Eigen::Vector3d& p, const Eigen::Vector3d& q)
  {
    Eigen::Vector3d d = (p - q).cwiseAbs();
    return d.maxCoeff();
  }

  static inline double norm(const Eigen::Vector3d& p)
  {
    return p.maxCoeff();
  }

  static inline double sqr(double r)
  {
    return r;
  }

  static inline double sqrt(double r)
  {
    return r;
  }
};

struct OctreeParams
{
 public:
  OctreeParams(uint32_t bucketSize = 32, bool copyPoints = false, double minExtent = 0.0f)
      : bucketSize(bucketSize), copyPoints(copyPoints), minExtent(minExtent)
  {
  }
  uint32_t bucketSize;
  bool copyPoints;
  double minExtent;
};

/** \brief Index-based Octree implementation offering different queries and insertion/removal of points.
 *
 * The index-based Octree uses a successor relation and a startIndex in each Octant to improve runtime
 * performance for radius queries. The efficient storage of the points by relinking list elements
 * bases on the insight that children of an Octant contain disjoint subsets of points inside the Octant and
 * that we can reorganize the points such that we get an continuous single connect list that we can use to
 * store in each octant the start of this list.
 *
 * Special about the implementation is that it allows to search for neighbors with arbitrary p-norms, which
 * distinguishes it from most other Octree implementations.
 *
 * We decided to implement the Octree using a template for points and containers. The container must have an
 * operator[], which allows to access the points, and a size() member function, which allows to get the size of the
 * container. For the points, we used an access trait to access the coordinates inspired by boost.geometry.
 * The implementation already provides a general access trait, which expects to have public member variables x,y,z.
 *
 * f you use the implementation or ideas from the corresponding paper in your academic work, it would be nice if you
 * cite the corresponding paper:
 *
 *    J. Behley, V. Steinhage, A.B. Cremers. Efficient Radius Neighbor Search in Three-dimensional Point Clouds,
 *    Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2015.
 *
 * In future, we might add also other neighbor queries and implement the removal and adding of points.
 *
 * \version 0.1-icra
 *
 * \author behley
 */

class Octree
{
 public:

  class Octant
  {
   public:
    inline Octant();
    inline ~Octant();

    bool isLeaf;

    // bounding box of the octant needed for overlap and contains tests...
    Eigen::Vector3d center;  // center
    double extent;           // half of side-length

    size_t start, end;  // start and end in succ_
    size_t size;        // number of points

    Octant* child[8];
  };

  inline Octree();
  inline ~Octree();

  /** \brief initialize octree with all points **/
  inline void initialize(const std::shared_ptr<Eigen::Matrix3Xd>& pts, const OctreeParams& params = OctreeParams());

  /** \brief initialize octree only from pts that are inside indexes. **/
  inline void initialize(const std::shared_ptr<Eigen::Matrix3Xd>& pts, const std::vector<size_t>& indexes,
                  const OctreeParams& params = OctreeParams());

  /** \brief remove all data inside the octree. **/
  inline void clear();

  /** \brief radius neighbor queries where radius determines the maximal radius of reported indices of points in
   * resultIndices **/
  template <typename Distance>
  void radiusNeighbors(const Eigen::Vector3d& query, double radius, std::vector<size_t>& resultIndices) const;

  /** \brief radius neighbor queries with explicit (squared) distance computation. **/
  template <typename Distance>
  void radiusNeighbors(const Eigen::Vector3d& query, double radius, std::vector<size_t>& resultIndices,
                       std::vector<double>& distances) const;

  /** \brief nearest neighbor queries. Using minDistance >= 0, we explicitly disallow self-matches.
   * @return index of nearest neighbor n with Distance::compute(query, n) > minDistance if return true
   **/
  template <typename Distance>
  bool findNeighbor(const Eigen::Vector3d& query, size_t& resultIndex, double minDistance = -1) const;
  
  inline const Octant* root() const
  {
    return root_;
  }

  inline const std::vector<size_t> successors() const
  {
    return successors_;
  }
 protected:
  

  // not copyable, not assignable ...
  Octree(Octree&);
  Octree& operator=(const Octree& oct);

  /**
   * \brief creation of an octant using the elements starting at startIdx.
   *
   * The method reorders the index such that all points are correctly linked to successors belonging
   * to the same octant.
   *
   * \param x,y,z           center coordinates of octant
   * \param extent          extent of octant
   * \param startIdx        first index of points inside octant
   * \param endIdx          last index of points inside octant
   * \param size            number of points in octant
   *
   * \return  octant with children nodes.
   */
  inline Octant* createOctant(const Eigen::Vector3d& center, double extent, size_t startIdx, size_t endIdx, size_t size);

  /** @return true, if search finished, otherwise false. **/
  template <typename Distance>
  bool findNeighbor(const Octant* octant, const Eigen::Vector3d& query, double minDistance, double& maxDistance,
                    size_t& resultIndex) const;

  template <typename Distance>
  void radiusNeighbors(const Octant* octant, const Eigen::Vector3d& query, double radius, double sqrRadius,
                       std::vector<size_t>& resultIndices) const;

  template <typename Distance>
  void radiusNeighbors(const Octant* octant, const Eigen::Vector3d& query, double radius, double sqrRadius,
                       std::vector<size_t>& resultIndices, std::vector<double>& distances) const;

  /** \brief test if search ball S(q,r) overlaps with octant
   *
   * @param query   query point
   * @param radius  "squared" radius
   * @param o       pointer to octant
   *
   * @return true, if search ball overlaps with octant, false otherwise.
   */
  template <typename Distance>
  static bool overlaps(const Eigen::Vector3d& query, double radius, double sqRadius, const Octant* o);

  /** \brief test if search ball S(q,r) contains octant
   *
   * @param query    query point
   * @param sqRadius "squared" radius
   * @param octant   pointer to octant
   *
   * @return true, if search ball overlaps with octant, false otherwise.
   */
  template <typename Distance>
  static bool contains(const Eigen::Vector3d& query, double sqRadius, const Octant* octant);

  /** \brief test if search ball S(q,r) is completely inside octant.
   *
   * @param query   query point
   * @param radius  radius r
   * @param octant  point to octant.
   *
   * @return true, if search ball is completely inside the octant, false otherwise.
   */
  template <typename Distance>
  static bool inside(const Eigen::Vector3d& query, double radius, const Octant* octant);

  OctreeParams params_;
  Octant* root_;
  std::shared_ptr<Eigen::Matrix3Xd> data_;

  std::vector<size_t> successors_;  // single connected list of next point indices...

  friend class ::OctreeTest;
};

Octree::Octant::Octant()
    : isLeaf(true), center(Eigen::Vector3d::Zero()), extent(0.0), start(0), end(0), size(0)
{
  memset(&child, 0, 8 * sizeof(Octant*));
}

Octree::Octant::~Octant()
{
  for (size_t i = 0; i < 8; ++i) delete child[i];
}

Octree::Octree()
    : root_(0)
{
}

Octree::~Octree()
{
  delete root_;
}

void Octree::initialize(const std::shared_ptr<Eigen::Matrix3Xd>& pts, const OctreeParams& params)
{
  clear();
  params_ = params;

  if (params.copyPoints)
  {
    data_ = std::shared_ptr<Eigen::Matrix3Xd>(new Eigen::Matrix3Xd());
    *data_ = *pts;
  }
  else
  {
    data_ = pts;
  }
  

  const size_t N = pts->cols();
  successors_ = std::vector<size_t>(N);
  std::iota(successors_.begin(), successors_.end(), 1.0);
  
  // determine axis-aligned bounding box.
  Eigen::Vector3d aabb_min = data_->rowwise().minCoeff();
  Eigen::Vector3d aabb_max = data_->rowwise().maxCoeff();

  Eigen::Vector3d aabb_range = (aabb_max - aabb_min) * 0.5;
  Eigen::Vector3d aabb_center = (aabb_max + aabb_min) * 0.5;

  root_ = createOctant(aabb_center, aabb_range.maxCoeff(), 0, N - 1, N);
}

void Octree::initialize(const std::shared_ptr<Eigen::Matrix3Xd>& pts, const std::vector<size_t>& indexes,
                                            const OctreeParams& params)
{
  clear();
  params_ = params;

  if (params.copyPoints)
  {
    data_ = std::shared_ptr<Eigen::Matrix3Xd>(new Eigen::Matrix3Xd());
    *data_ = *pts;
  }
  else
  {
    data_ = pts;
  }

  const size_t N = data_->cols();
  successors_ = std::vector<size_t>(N);

  if (indexes.size() == 0) return;

  // determine axis-aligned bounding box.
  size_t lastIdx = indexes[0];
  Eigen::Vector3d aabb_min;
  Eigen::Vector3d aabb_max;

  aabb_min[0] = data_->col(lastIdx)[0];
  aabb_min[1] = data_->col(lastIdx)[1];
  aabb_min[2] = data_->col(lastIdx)[2];
  aabb_max = aabb_min;

  for (size_t i = 1; i < indexes.size(); ++i)
  {
    size_t idx = indexes[i];
    // initially each element links simply to the following element.
    successors_[lastIdx] = idx;

    const Eigen::Vector3d& p = data_->col(idx);

    if (p[0] < aabb_min[0]) aabb_min[0] = p[0];
    if (p[1] < aabb_min[1]) aabb_min[1] = p[1];
    if (p[2] < aabb_min[2]) aabb_min[2] = p[2];
    if (p[0] > aabb_max[0]) aabb_max[0] = p[0];
    if (p[1] > aabb_max[1]) aabb_max[1] = p[1];
    if (p[2] > aabb_max[2]) aabb_max[2] = p[2];

    lastIdx = idx;
  }

  Eigen::Vector3d aabb_range = (aabb_max - aabb_min) * 0.5;
  Eigen::Vector3d aabb_center = (aabb_max + aabb_min) * 0.5;

  root_ = createOctant(aabb_center, aabb_range.maxCoeff(), indexes[0], lastIdx, indexes.size());
}

void Octree::clear()
{
  delete root_;
  root_ = 0;
  data_.reset();
  successors_.clear();
}

typename Octree::Octant* Octree::createOctant(const Eigen::Vector3d& center,
                                              double extent, size_t startIdx,
                                              size_t endIdx, size_t size)
{
  // For a leaf we don't have to change anything; points are already correctly linked or correctly reordered.
  Octant* octant = new Octant;

  octant->isLeaf = true;

  octant->center = center;
  octant->extent = extent;

  octant->start = startIdx;
  octant->end = endIdx;
  octant->size = size;

  static const double factor[] = {-0.5f, 0.5f};

  // subdivide subset of points and re-link points according to Morton codes
  if (size > params_.bucketSize && extent > 2 * params_.minExtent)
  {
    octant->isLeaf = false;

    std::vector<size_t> childStarts(8, 0);
    std::vector<size_t> childEnds(8, 0);
    std::vector<size_t> childSizes(8, 0);

    // re-link disjoint child subsets...
    size_t idx = startIdx;

    for (size_t i = 0; i < size; ++i)
    {
      const Eigen::Vector3d& p = data_->col(idx);

      // determine Morton code for each point...
      size_t mortonCode = 0;
      if (p[0] > center[0]) mortonCode |= 1;
      if (p[1] > center[1]) mortonCode |= 2;
      if (p[2] > center[2]) mortonCode |= 4;

      // set child starts and update successors...
      if (childSizes[mortonCode] == 0)
        childStarts[mortonCode] = idx;
      else
        successors_[childEnds[mortonCode]] = idx;
      childSizes[mortonCode] += 1;

      childEnds[mortonCode] = idx;
      idx = successors_[idx];
    }

    // now, we can create the child nodes...
    double childExtent = 0.5f * extent;
    bool firsttime = true;
    size_t lastChildIdx = 0;
    for (size_t i = 0; i < 8; ++i)
    {
      if (childSizes[i] == 0) continue;

      Eigen::Vector3d child_center(center[0] + factor[(i & 1) > 0] * extent,
                                   center[1] + factor[(i & 2) > 0] * extent,
                                   center[2] + factor[(i & 4) > 0] * extent);

      octant->child[i] = createOctant(child_center, childExtent, childStarts[i], childEnds[i], childSizes[i]);

      if (firsttime)
        octant->start = octant->child[i]->start;
      else
        successors_[octant->child[lastChildIdx]->end] =
            octant->child[i]->start;  // we have to ensure that also the child ends link to the next child start.

      lastChildIdx = i;
      octant->end = octant->child[i]->end;
      firsttime = false;
    }
  }

  return octant;
}

template <typename Distance>
void Octree::radiusNeighbors(const Octant* octant, const Eigen::Vector3d& query, double radius,
                             double sqrRadius, std::vector<size_t>& resultIndices) const
{

  // if search ball S(q,r) contains octant, simply add point indexes.
  if (contains<Distance>(query, sqrRadius, octant))
  {
    size_t idx = octant->start;
    for (size_t i = 0; i < octant->size; ++i)
    {
      resultIndices.push_back(idx);
      idx = successors_[idx];
    }

    return;  // early pruning.
  }

  if (octant->isLeaf)
  {
    size_t idx = octant->start;
    for (size_t i = 0; i < octant->size; ++i)
    {
      const Eigen::Vector3d& p = data_->col(idx);
      double dist = Distance::compute(query, p);
      if (dist < sqrRadius) resultIndices.push_back(idx);
      idx = successors_[idx];
    }

    return;
  }

  // check whether child nodes are in range.
  for (size_t c = 0; c < 8; ++c)
  {
    if (octant->child[c] == 0) continue;
    if (!overlaps<Distance>(query, radius, sqrRadius, octant->child[c])) continue;
    radiusNeighbors<Distance>(octant->child[c], query, radius, sqrRadius, resultIndices);
  }
}

template <typename Distance>
void Octree::radiusNeighbors(const Octant* octant, const Eigen::Vector3d& query, double radius,
                             double sqrRadius, std::vector<size_t>& resultIndices,
                             std::vector<double>& distances) const
{
  // if search ball S(q,r) contains octant, simply add point indexes and compute squared distances.
  if (contains<Distance>(query, sqrRadius, octant))
  {
    size_t idx = octant->start;
    for (size_t i = 0; i < octant->size; ++i)
    {
      resultIndices.push_back(idx);
      distances.push_back(Distance::compute(query, data_->col(idx)));
      idx = successors_[idx];
    }

    return;  // early pruning.
  }

  if (octant->isLeaf)
  {
    size_t idx = octant->start;
    for (size_t i = 0; i < octant->size; ++i)
    {
      const Eigen::Vector3d& p = data_->col(idx);
      double dist = Distance::compute(query, p);
      if (dist < sqrRadius)
      {
        resultIndices.push_back(idx);
        distances.push_back(dist);
      }
      idx = successors_[idx];
    }

    return;
  }

  // check whether child nodes are in range.
  for (size_t c = 0; c < 8; ++c)
  {
    if (octant->child[c] == 0) continue;
    if (!overlaps<Distance>(query, radius, sqrRadius, octant->child[c])) continue;
    radiusNeighbors<Distance>(octant->child[c], query, radius, sqrRadius, resultIndices, distances);
  }
}

template <typename Distance>
void Octree::radiusNeighbors(const Eigen::Vector3d& query, double radius,
                             std::vector<size_t>& resultIndices) const
{
  resultIndices.clear();
  if (root_ == 0) return;

  double sqrRadius = Distance::sqr(radius);  // "squared" radius
  radiusNeighbors<Distance>(root_, query, radius, sqrRadius, resultIndices);
}

template <typename Distance>
void Octree::radiusNeighbors(const Eigen::Vector3d& query, double radius,
                             std::vector<size_t>& resultIndices,
                             std::vector<double>& distances) const
{
  resultIndices.clear();
  distances.clear();
  if (root_ == 0) return;

  double sqrRadius = Distance::sqr(radius);  // "squared" radius
  radiusNeighbors<Distance>(root_, query, radius, sqrRadius, resultIndices, distances);
}

template <typename Distance>
bool Octree::overlaps(const Eigen::Vector3d& query, double radius, double sqRadius, const Octant* o)
{
  // we exploit the symmetry to reduce the test to testing if its inside the Minkowski sum around the positive quadrant.
  Eigen::Vector3d d = (query - o->center).cwiseAbs();

  double maxdist = radius + o->extent;

  // Completely outside, since q' is outside the relevant area.
  if (d[0] > maxdist || d[1] > maxdist || d[2] > maxdist) return false;

  int32_t num_less_extent = (d[0] < o->extent) + (d[1] < o->extent) + (d[2] < o->extent);

  // Checking different cases:

  // a. inside the surface region of the octant.
  if (num_less_extent > 1) return true;

  // b. checking the corner region && edge region.
  d[0] = std::max(d[0] - o->extent, 0.0);
  d[1] = std::max(d[1] - o->extent, 0.0);
  d[2] = std::max(d[2] - o->extent, 0.0);

  return (Distance::norm(d) < sqRadius);
}

template <typename Distance>
bool Octree::contains(const Eigen::Vector3d& query, double sqRadius, const Octant* o)
{
  // we exploit the symmetry to reduce the test to test
  // whether the farthest corner is inside the search ball.
  Eigen::Vector3d d = (query - o->center).cwiseAbs();
  // reminder: (x, y, z) - (-e, -e, -e) = (x, y, z) + (e, e, e)
  d = d.array() + o->extent;
  
  return (Distance::norm(d) < sqRadius);
}

template <typename Distance>
bool Octree::findNeighbor(const Eigen::Vector3d& query, size_t& resultIndex, double minDistance) const
{
  double maxDistance = std::numeric_limits<double>::infinity();
  return findNeighbor<Distance>(root_, query, minDistance, maxDistance, resultIndex);
}

template <typename Distance>
bool Octree::findNeighbor(const Octant* octant, const Eigen::Vector3d& query, double minDistance,
                          double& maxDistance, size_t& resultIndex) const
{
  // 1. first descend to leaf and check in leafs points.
  if (octant->isLeaf)
  {
    size_t idx = octant->start;
    double sqrMaxDistance = Distance::sqr(maxDistance);
    double sqrMinDistance = (minDistance < 0) ? minDistance : Distance::sqr(minDistance);

    for (size_t i = 0; i < octant->size; ++i)
    {
      const Eigen::Vector3d& p = data_->col(idx);
      double dist = Distance::compute(query, p);
      if (dist > sqrMinDistance && dist < sqrMaxDistance)
      {
        resultIndex = idx;
        sqrMaxDistance = dist;
      }
      idx = successors_[idx];
    }

    maxDistance = Distance::sqrt(sqrMaxDistance);
    return inside<Distance>(query, maxDistance, octant);
  }

  // determine Morton code for each point...
  size_t mortonCode = 0;
  if (query[0] > octant->center[0]) mortonCode |= 1;
  if (query[1] > octant->center[1]) mortonCode |= 2;
  if (query[2] > octant->center[2]) mortonCode |= 4;

  if (octant->child[mortonCode] != 0)
  {
    if (findNeighbor<Distance>(octant->child[mortonCode], query, minDistance, maxDistance, resultIndex)) return true;
  }

  // 2. if current best point completely inside, just return.
  double sqrMaxDistance = Distance::sqr(maxDistance);

  // 3. check adjacent octants for overlap and check these if necessary.
  for (size_t c = 0; c < 8; ++c)
  {
    if (c == mortonCode) continue;
    if (octant->child[c] == 0) continue;
    if (!overlaps<Distance>(query, maxDistance, sqrMaxDistance, octant->child[c])) continue;
    if (findNeighbor<Distance>(octant->child[c], query, minDistance, maxDistance, resultIndex))
      return true;  // early pruning
  }

  // all children have been checked...check if point is inside the current octant...
  return inside<Distance>(query, maxDistance, octant);
}

template <typename Distance>
bool Octree::inside(const Eigen::Vector3d& query, double radius, const Octant* octant)
{
  // we exploit the symmetry to reduce the test to test
  // whether the farthest corner is inside the search ball.
  Eigen::Vector3d d = (query - octant->center).cwiseAbs();
  d = d.array() + radius;
  
  if (d[0] > octant->extent) return false;
  if (d[1] > octant->extent) return false;
  if (d[2] > octant->extent) return false;

  return true;
}
}

#endif /* OCTREE_HPP_ */
