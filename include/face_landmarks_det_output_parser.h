// Copyright (c) 2024，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FACE_LANDMARKS_DET_OUTPUT_PARSER_H
#define FACE_LANDMARKS_DET_OUTPUT_PARSER_H

#include "rclcpp/rclcpp.hpp"
#include "dnn_node/dnn_node_data.h"

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::Model;

/**
 * @brief 2D point coordinates with score
 */
template <typename Dtype> struct Point_
{
    inline Point_()
    {
    }
    inline Point_(Dtype x_, Dtype y_, float score_ = 0.0) : x(x_), y(y_), score(score_)
    {
    }

    Dtype x = 0;
    Dtype y = 0;
    float score = 0.0;
};
typedef Point_<float> Point;
typedef std::vector<Point> Landmarks;

/**
 * @brief face landmarks detection result
 */
class FaceLandmarksDetResult
{
public:
    std::vector<Landmarks> values;

    void Reset()
    {
        values.clear();
    }
};

/**
 * @brief face landmarks detection output parser process
 */
class FaceLandmarksDetOutputParser
{
public:
    FaceLandmarksDetOutputParser(const rclcpp::Logger &logger);
    ~FaceLandmarksDetOutputParser() = default;

    // 对于roi infer task，每个roi对应一次Parse
    // 因此需要在Parse中实现output和roi的match处理，即当前的Parse对应的是那个roi
    int32_t Parse(std::shared_ptr<FaceLandmarksDetResult> &output, std::vector<std::shared_ptr<DNNTensor>> &output_tensors, std::shared_ptr<std::vector<hbDNNRoi>> rois);

private:
    float quanti_shift(int32_t data, uint32_t shift)
    {
        return static_cast<float>(data) / static_cast<float>(1 << shift);
    }

    float quanti_scale(int32_t data, float scale)
    {
        return data * scale;
    }

    // log print
    rclcpp::Logger logger_;
};

#endif