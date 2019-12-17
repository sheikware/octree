#include <gtest/gtest.h>
#include <boost/random.hpp>
#include <map>
#include <queue>
#include <string>

#include "Octree.hpp"

namespace
{

class NaiveNeighborSearch
{
 public:
  void initialize(const Eigen::Matrix3Xd& points)
  {
    data_ = points;
  }

  template <typename Distance>
  bool findNeighbor(const Eigen::Vector3d& query, size_t& resultIndex, double minDistance = -1.0)
  {
    if (data_.cols() == 0) return false;

    double maxDistance = std::numeric_limits<double>::infinity();
    double sqrMinDistance = (minDistance < 0) ? minDistance : Distance::sqr(minDistance);
    resultIndex = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < data_.cols(); ++i)
    {
      double dist = Distance::compute(query, data_.col(i));
      if ((dist > sqrMinDistance) && (dist < maxDistance))
      {
        maxDistance = dist;
        resultIndex = i;
      }
    }

    return true;
  }

  template <typename Distance>
  void radiusNeighbors(const Eigen::Vector3d& query, double radius, std::vector<size_t>& resultIndices)
  {
    resultIndices.clear();
    double sqrRadius = Distance::sqr(radius);

    for (size_t i = 0; i < data_.cols(); ++i)
    {
      if (Distance::compute(query, data_.col(i)) < sqrRadius)
      {
        resultIndices.push_back(i);
      }
    }
  }

 protected:
  Eigen::Matrix3Xd data_;
};

// The fixture for testing class Foo.
class OctreeTest : public ::testing::Test
{
 public:
  typedef unibn::Octree::Octant Octant;

 protected:
  // helper methods to access the protected parts of octree for consistency
  // checks.
  const typename unibn::Octree::Octant* getRoot(const unibn::Octree& oct)
  {
    return oct.root_;
  }

  const std::vector<size_t>& getSuccessors(const unibn::Octree& oct)
  {
    return oct.successors_;
  }

  template <typename Distance>
  bool overlaps(const Eigen::Vector3d& query, double radius, double sqRadius, const Octant* o)
  {
    return unibn::Octree::template overlaps<Distance>(query, radius, sqRadius, o);
  }
};

void randomPoints(Eigen::Matrix3Xd& pts, size_t N, uint32_t seed = 0)
{
  boost::mt11213b mtwister(seed);
  boost::uniform_01<> gen;
  pts.resize(3, N);
  // generate N random points in [-5.0,5.0] x [-5.0,5.0] x [-5.0,5.0]...
  for (size_t i = 0; i < N; ++i)
  {
    Eigen::Vector3d p(10.0 * gen(mtwister) - 5.0, 10.0 * gen(mtwister) - 5.0, 10.0 * gen(mtwister) - 5.0);
    pts.col(i) = p;
  }
}

TEST_F(OctreeTest, Initialize)
{

  size_t N = 1000;
  unibn::OctreeParams params;
  params.bucketSize = 16;

  unibn::Octree oct;

  const Octant* root = getRoot(oct);
  const std::vector<size_t>& successors = getSuccessors(oct);

  ASSERT_EQ(0, root);

  Eigen::Matrix3Xd points;
  randomPoints(points, N, 1337);

  oct.initialize(points, params);

  root = getRoot(oct);

  // check first some pre-requisits.
  ASSERT_EQ(true, (root != 0));
  ASSERT_EQ(N, successors.size());

  std::vector<size_t> elementCount(N, 0);
  size_t idx = root->start;
  for (size_t i = 0; i < N; ++i)
  {
    ASSERT_LT(idx, N);
    ASSERT_LE(successors[idx], N);
    elementCount[idx] += 1;
    ASSERT_EQ(1, elementCount[idx]);
    idx = successors[idx];
  }

  // check that each index was found.
  for (size_t i = 0; i < N; ++i)
  {
    ASSERT_EQ(1, elementCount[i]);
  }

  // test if each Octant contains only points inside the octant and child
  // octants have only real subsets of parents!
  std::queue<const Octant*> queue;
  queue.push(root);
  std::vector<size_t> assignment(N, std::numeric_limits<size_t>::max());

  while (!queue.empty())
  {
    const Octant* octant = queue.front();
    queue.pop();

    // check points.
    ASSERT_LT(octant->start, N);

    // test if each point assigned to a octant really is inside the octant.

    size_t idx = octant->start;
    size_t lastIdx = octant->start;
    for (size_t i = 0; i < octant->size; ++i)
    {
      Eigen::Vector3d p = points.col(idx) - octant->center;

      ASSERT_LE(std::abs(p[0]), octant->extent);
      ASSERT_LE(std::abs(p[1]), octant->extent);
      ASSERT_LE(std::abs(p[2]), octant->extent);
      assignment[idx] = std::numeric_limits<size_t>::max();  // reset of child assignments.
      lastIdx = idx;
      idx = successors[idx];
    }
    ASSERT_EQ(octant->end, lastIdx);

    bool shouldBeLeaf = true;
    Octant* firstchild = 0;
    Octant* lastchild = 0;
    size_t pointSum = 0;

    for (size_t c = 0; c < 8; ++c)
    {
      Octant* child = octant->child[c];
      if (child == 0) continue;
      shouldBeLeaf = false;

      // child nodes should have start end intervals, which are true subsets of
      // the parent.
      if (firstchild == 0) firstchild = child;
      // the child nodes should have intervals, where succ(e_{c-1}) == s_{c},
      // and \sum_c size(c) = parent size!
      if (lastchild != 0) ASSERT_EQ(child->start, successors[lastchild->end]);

      pointSum += child->size;
      lastchild = child;
      size_t idx = child->start;
      for (size_t i = 0; i < child->size; ++i)
      {
        // check if points are uniquely assigned to single child octant.
        ASSERT_EQ(std::numeric_limits<size_t>::max(), assignment[idx]);
        assignment[idx] = c;
        idx = successors[idx];
      }

      queue.push(child);
    }

    // consistent start/end of octant and its first and last children.
    if (firstchild != 0) ASSERT_EQ(octant->start, firstchild->start);
    if (lastchild != 0) ASSERT_EQ(octant->end, lastchild->end);

    // check leafs flag.
    ASSERT_EQ(shouldBeLeaf, octant->isLeaf);
    ASSERT_EQ((octant->size <= params.bucketSize), octant->isLeaf);

    // test if every point is assigned to a child octant.
    if (!octant->isLeaf)
    {
      ASSERT_EQ(octant->size, pointSum);
      size_t idx = octant->start;
      for (size_t i = 0; i < octant->size; ++i)
      {
        ASSERT_LT(assignment[idx], std::numeric_limits<size_t>::max());
        idx = successors[idx];
      }
    }
  }
}

TEST_F(OctreeTest, Initialize_minExtent)
{

  size_t N = 1000;
  unibn::OctreeParams params;
  params.bucketSize = 16;
  params.minExtent = 1.0f;

  unibn::Octree oct;

  const Octant* root = getRoot(oct);
  const std::vector<size_t>& successors = getSuccessors(oct);

  ASSERT_EQ(0, root);

  Eigen::Matrix3Xd points;
  randomPoints(points, N, 1337);

  oct.initialize(points, params);

  root = getRoot(oct);

  // check first some pre-requisits.
  ASSERT_EQ(true, (root != 0));
  ASSERT_EQ(N, successors.size());

  std::vector<size_t> elementCount(N, 0);
  size_t idx = root->start;
  for (size_t i = 0; i < N; ++i)
  {
    ASSERT_LT(idx, N);
    ASSERT_LE(successors[idx], N);
    elementCount[idx] += 1;
    ASSERT_EQ(1, elementCount[idx]);
    idx = successors[idx];
  }

  // check that each index was found.
  for (size_t i = 0; i < N; ++i)
  {
    ASSERT_EQ(1, elementCount[i]);
  }

  // test if each Octant contains only points inside the octant and child
  // octants have only real subsets of parents!
  std::queue<const Octant*> queue;
  queue.push(root);
  std::vector<size_t> assignment(N, std::numeric_limits<size_t>::max());

  while (!queue.empty())
  {
    const Octant* octant = queue.front();
    queue.pop();

    // check points.
    ASSERT_LT(octant->start, N);

    // test if each point assigned to a octant really is inside the octant.

    size_t idx = octant->start;
    size_t lastIdx = octant->start;
    for (size_t i = 0; i < octant->size; ++i)
    {
      Eigen::Vector3d p = points.col(idx) - octant->center;

      ASSERT_LE(std::abs(p[0]), octant->extent);
      ASSERT_LE(std::abs(p[1]), octant->extent);
      ASSERT_LE(std::abs(p[2]), octant->extent);
      assignment[idx] = std::numeric_limits<size_t>::max();  // reset of child assignments.
      lastIdx = idx;
      idx = successors[idx];
    }
    ASSERT_EQ(octant->end, lastIdx);

    bool shouldBeLeaf = true;
    Octant* firstchild = 0;
    Octant* lastchild = 0;
    size_t pointSum = 0;

    for (size_t c = 0; c < 8; ++c)
    {
      Octant* child = octant->child[c];
      if (child == 0) continue;
      shouldBeLeaf = false;

      // child nodes should have start end intervals, which are true subsets of
      // the parent.
      if (firstchild == 0) firstchild = child;
      // the child nodes should have intervals, where succ(e_{c-1}) == s_{c},
      // and \sum_c size(c) = parent size!
      if (lastchild != 0) ASSERT_EQ(child->start, successors[lastchild->end]);

      pointSum += child->size;
      lastchild = child;
      size_t idx = child->start;
      for (size_t i = 0; i < child->size; ++i)
      {
        // check if points are uniquely assigned to single child octant.
        ASSERT_EQ(std::numeric_limits<size_t>::max(), assignment[idx]);
        assignment[idx] = c;
        idx = successors[idx];
      }

      queue.push(child);
    }

    // consistent start/end of octant and its first and last children.
    if (firstchild != 0) ASSERT_EQ(octant->start, firstchild->start);
    if (lastchild != 0) ASSERT_EQ(octant->end, lastchild->end);

    // check leafs flag.
    ASSERT_EQ(shouldBeLeaf, octant->isLeaf);
    ASSERT_EQ((octant->size <= params.bucketSize || octant->extent < 2.0f * params.minExtent), octant->isLeaf);
    ASSERT_GE(octant->extent, params.minExtent);

    // test if every point is assigned to a child octant.
    if (!octant->isLeaf)
    {
      ASSERT_EQ(octant->size, pointSum);
      size_t idx = octant->start;
      for (size_t i = 0; i < octant->size; ++i)
      {
        ASSERT_LT(assignment[idx], std::numeric_limits<size_t>::max());
        idx = successors[idx];
      }
    }
  }
}

TEST_F(OctreeTest, FindNeighbor)
{
  // compare with bruteforce search.
  size_t N = 1000;

  boost::mt11213b mtwister(1234);
  boost::uniform_int<> uni_dist(0, N - 1);

  Eigen::Matrix3Xd points;
  randomPoints(points, N, 1234);

  NaiveNeighborSearch bruteforce;
  bruteforce.initialize(points);
  unibn::Octree octree;
  octree.initialize(points);

  for (size_t i = 0; i < 10; ++i)
  {
    size_t index = uni_dist(mtwister);
    const Eigen::Vector3d& query = points.col(index);

    // allow self-match
    size_t brute_result;
    bruteforce.findNeighbor<unibn::L2Distance>(query, brute_result);
    ASSERT_EQ(index, brute_result);

    size_t octree_result;
    octree.findNeighbor<unibn::L2Distance>(query, octree_result);
    ASSERT_EQ(brute_result, octree_result);

    // disallow self-match
    size_t bfneighbor;
    bruteforce.findNeighbor<unibn::L2Distance>(query, bfneighbor, 0.3);
    size_t octneighbor;
    octree.findNeighbor<unibn::L2Distance>(query, octneighbor, 0.3);

    ASSERT_EQ(bfneighbor, octneighbor);
  }
}

template <typename T>
bool similarVectors(std::vector<T>& vec1, std::vector<T>& vec2)
{
  if (vec1.size() != vec2.size())
  {
    std::cout << "expected size = " << vec1.size() << ", but got size = " << vec2.size() << std::endl;
    return false;
  }

  for (uint32_t i = 0; i < vec1.size(); ++i)
  {
    bool found = false;
    for (uint32_t j = 0; j < vec2.size(); ++j)
    {
      if (vec1[i] == vec2[j])
      {
        found = true;
        break;
      }
    }
    if (!found)
    {
      std::cout << i << "-th element (" << vec1[i] << ") not found." << std::endl;
      return false;
    }
  }

  return true;
}

TEST_F(OctreeTest, RadiusNeighbors)
{
  size_t N = 1000;

  boost::mt11213b mtwister(1234);
  boost::uniform_int<> uni_dist(0, N - 1);

  Eigen::Matrix3Xd points;
  randomPoints(points, N, 1234);

  NaiveNeighborSearch bruteforce;
  bruteforce.initialize(points);
  unibn::Octree octree;
  octree.initialize(points);

  double radii[4] = {0.5, 1.0, 2.0, 5.0};

  for (size_t r = 0; r < 4; ++r)
  {
    for (size_t i = 0; i < 10; ++i)
    {
      std::vector<size_t> neighborsBruteforce;
      std::vector<size_t> neighborsOctree;

      const Eigen::Vector3d& query = points.col(uni_dist(mtwister));

      bruteforce.radiusNeighbors<unibn::L2Distance>(query, radii[r], neighborsBruteforce);
      octree.radiusNeighbors<unibn::L2Distance>(query, radii[r], neighborsOctree);
      ASSERT_EQ(true, similarVectors(neighborsBruteforce, neighborsOctree));

      bruteforce.radiusNeighbors<unibn::L1Distance>(query, radii[r], neighborsBruteforce);
      octree.radiusNeighbors<unibn::L1Distance>(query, radii[r], neighborsOctree);

      ASSERT_EQ(true, similarVectors(neighborsBruteforce, neighborsOctree));

      bruteforce.radiusNeighbors<unibn::MaxDistance>(query, radii[r], neighborsBruteforce);
      octree.radiusNeighbors<unibn::MaxDistance>(query, radii[r], neighborsOctree);

      ASSERT_EQ(true, similarVectors(neighborsBruteforce, neighborsOctree));
    }
  }
}

TEST_F(OctreeTest, OverlapTest)
{
  Octant octant;

  octant.center = Eigen::Vector3d(1.0, 1.0, 1.0);
  octant.extent = 0.5;

  // completely inside
  Eigen::Vector3d query(1.25, 1.25, 0.5);
  double radius = 1.0;

  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  // faces of octant.
  query = Eigen::Vector3d(1.75, 1.0, 1.0);
  radius = 0.5;

  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.0, 1.75, 1.0);
  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.0, 1.0, 1.75);
  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.0, 1.0, 2.75);
  ASSERT_FALSE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  // Edge cases:
  query = Eigen::Vector3d(1.65, 1.65, 1.25);
  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.25, 1.65, 1.65);
  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.65, 1.25, 1.75);
  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.9, 1.25, 1.9);
  ASSERT_FALSE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.25, 1.9, 1.9);
  ASSERT_FALSE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.9, 1.9, 1.25);
  ASSERT_FALSE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  // corner cases:
  query = Eigen::Vector3d(1.65, 1.65, 1.65);
  ASSERT_TRUE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  query = Eigen::Vector3d(1.95, 1.95, 1.95);
  ASSERT_FALSE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));

  // edge special case, see Issue #3 -- Edge
  octant.center = Eigen::Vector3d(0.025, -0.025, -0.025);
  octant.extent = 0.025;

  query = Eigen::Vector3d(0.025, 0.025, 0.025);
  radius = 0.025;

  ASSERT_FALSE(overlaps<unibn::L2Distance>(query, radius, radius * radius, &octant));
}
}  // namespace

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
