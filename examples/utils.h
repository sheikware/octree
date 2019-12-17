#ifndef EXAMPLES_UTILS_H_
#define EXAMPLES_UTILS_H_

#include <fstream>
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <vector>
#include <Eigen/Core>

void readPoints(const std::string& filename, Eigen::Matrix3Xd& points)
{
  std::ifstream in(filename.c_str());
  std::string line;
  boost::char_separator<char> sep(" ");
  // read point cloud from "freiburg format"
  
  std::vector<Eigen::Vector3d> pts;
  
  while (!in.eof())
  {
    std::getline(in, line);
    in.peek();

    boost::tokenizer<boost::char_separator<char> > tokenizer(line, sep);
    std::vector<std::string> tokens(tokenizer.begin(), tokenizer.end());

    if (tokens.size() != 6) continue;
    double x = boost::lexical_cast<double>(tokens[3]);
    double y = boost::lexical_cast<double>(tokens[4]);
    double z = boost::lexical_cast<double>(tokens[5]);

    pts.push_back(Eigen::Vector3d(x, y, z));
  }

  points.resize(3, pts.size());
  for (size_t i = 0; i < pts.size(); i++)
  {
    points.col(i) = pts[i];
  }
  in.close();
}

#endif /* EXAMPLES_UTILS_H_ */
