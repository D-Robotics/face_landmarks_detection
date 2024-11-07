#include "face_landmarks_det_output_parser.h"

FaceLandmarksDetOutputParser::FaceLandmarksDetOutputParser(const rclcpp::Logger &logger) : logger_(logger)
{
}

int32_t FaceLandmarksDetOutputParser::Parse(std::shared_ptr<FaceLandmarksDetResult> &output, std::vector<std::shared_ptr<DNNTensor>> &output_tensors, std::shared_ptr<std::vector<hbDNNRoi>> rois)
{
    RCLCPP_INFO(logger_, "=> output_tensors.size(): %ld", output_tensors.size());
    // check roi
    if (rois == nullptr || static_cast<int>(rois->size()) == 0)
    {
        RCLCPP_ERROR(logger_, "=> get null rois");
        return -1;
    }

    // allocate mem for result
    std::shared_ptr<FaceLandmarksDetResult> face_landmarks_det_result = nullptr;
    if (output == nullptr)
    {
        face_landmarks_det_result = std::make_shared<FaceLandmarksDetResult>();
        face_landmarks_det_result->Reset();
        output = face_landmarks_det_result;
    }
    else
    {
        face_landmarks_det_result = std::dynamic_pointer_cast<FaceLandmarksDetResult>(output);
        face_landmarks_det_result->Reset();
    }

    // parse output
    if (output_tensors[0]->properties.tensorType == HB_DNN_TENSOR_TYPE_S32 && output_tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC &&
        output_tensors[1]->properties.tensorType == HB_DNN_TENSOR_TYPE_S32 && output_tensors[1]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC)
    {
        // axis=0 calculates the x-direction coordinates, axis=1 calculates the y-direction coordinates
        for (int axis = 0; axis < 2; axis++)
        {
            RCLCPP_INFO(logger_, "=> parse ouput_tensor[%d]", axis);
            auto output_tensor = output_tensors[axis];
            int32_t *data = reinterpret_cast<int32_t *>(output_tensor->sysMem[0].virAddr);
            int *shape = output_tensor->properties.validShape.dimensionSize;
            int *aligned_shape = output_tensor->properties.alignedShape.dimensionSize;

            int align_b_step = aligned_shape[1] * aligned_shape[2] * aligned_shape[3];
            int align_h_step = aligned_shape[2] * aligned_shape[3];
            int align_w_step = aligned_shape[3];

            RCLCPP_INFO(logger_, "=> shape: [%d, %d, %d, %d]", shape[0], shape[1], shape[2], shape[3]);
            RCLCPP_INFO(logger_, "=> aligned_shape: [%d, %d, %d, %d]", aligned_shape[0], aligned_shape[1], aligned_shape[2], aligned_shape[3]);
            RCLCPP_INFO(logger_, "=> output_tensor->properties.shift.shiftLen: %d", output_tensor->properties.shift.shiftLen);

            // get landmarks num, generally 106 points
            int landmarks_num = shape[3];

            // get vector size, usually the ROI will be divided into 32 parts
            int vector_size = -1;
            if (axis == 0)
            {
                vector_size = shape[2];
            }
            else
            {
                vector_size = shape[1];
            }
            if (vector_size <= 1)
            {
                RCLCPP_ERROR(logger_, "=> get vector_size error: %d", vector_size);
                return -1;
            }
            RCLCPP_INFO(logger_, "=> landmarks_num: %d, vector_size: %d", landmarks_num, vector_size);

            // allocate memory for the result
            if (face_landmarks_det_result->values.empty())
            {
                face_landmarks_det_result->values.resize(shape[0]);
                for (size_t i = 0; i < face_landmarks_det_result->values.size(); i++)
                {
                    face_landmarks_det_result->values[i].resize(landmarks_num);
                }
                RCLCPP_INFO(logger_, "=> allocate memory for the result, face_landmarks_det_result->values.size(): %ld, face_landmarks_det_result->values[0].size(): %ld",
                            face_landmarks_det_result->values.size(), face_landmarks_det_result->values[0].size());
            }

            // batch corresponds to different roi areas
            for (int b = 0; b < shape[0]; b++)
            {
                // calc roi size
                auto roi = rois->at(b);
                float roi_width = roi.right - roi.left + 1;
                float roi_height = roi.bottom - roi.top + 1;
                for (int point_idx = 0; point_idx < landmarks_num; ++point_idx)
                {
                    // max_index is the coordinates of the face landmarks
                    float max_value = 0;
                    float max_index = 0;
                    for (int i = 0; i < vector_size; ++i)
                    {
                        int index = -1;
                        if (axis == 0)
                        {
                            index = b * align_b_step + i * align_w_step + point_idx;
                        }
                        else
                        {
                            index = b * align_b_step + i * align_h_step + point_idx;
                        }
                        if (index == -1)
                        {
                            RCLCPP_ERROR(logger_, "=> get index error");
                            return -1;
                        }

                        float value = 0;
                        if (output_tensor->properties.quantiType == SHIFT)
                        {
                            RCLCPP_INFO_ONCE(logger_, "=> quant shift");
                            value = quanti_shift(data[index], output_tensor->properties.shift.shiftData[point_idx]);
                        }
                        else if (output_tensor->properties.quantiType == SCALE)
                        {
                            RCLCPP_INFO_ONCE(logger_, "=> quant scale");
                            value = quanti_scale(data[index], output_tensor->properties.scale.scaleData[point_idx]);
                        }
                        else if (output_tensor->properties.quantiType == NONE)
                        {
                            RCLCPP_INFO_ONCE(logger_, "=> quant none");
                            value = data[index];
                        }
                        else
                        {
                            return -1;
                        }

                        if (value > max_value)
                        {
                            max_value = value;
                            max_index = i;
                        }
                    }
                    // RCLCPP_INFO(logger_, "=> point_idx: %d, max_value: %f, max_index: %d", point_idx, max_value, max_index);

                    /*
                     * It is necessary to determine whether the coordinate is left or right according to the left and right values of the max_value,
                     * and perform subpixel interpolation on the coordinates
                     */
                    float diff = 0;
                    if (max_index > 0 && max_index < vector_size - 1)
                    {
                        int left = -1;
                        int right = -1;
                        if (axis == 0)
                        {
                            left = b * align_b_step + (max_index - 1) * align_w_step + point_idx;
                            right = b * align_b_step + (max_index + 1) * align_w_step + point_idx;
                        }
                        else
                        {
                            left = b * align_b_step + (max_index - 1) * align_h_step + point_idx;
                            right = b * align_b_step + (max_index + 1) * align_h_step + point_idx;
                        }
                        if (left == -1 || right == -1)
                        {
                            RCLCPP_ERROR(logger_, "=> get index error");
                            return -1;
                        }

                        float left_val = -1;
                        float right_val = -1;
                        if (output_tensor->properties.quantiType == SHIFT)
                        {
                            left_val = quanti_shift(data[left], output_tensor->properties.shift.shiftData[point_idx]);
                            right_val = quanti_shift(data[right], output_tensor->properties.shift.shiftData[point_idx]);
                        }
                        else if (output_tensor->properties.quantiType == SCALE)
                        {
                            left_val = quanti_scale(data[left], output_tensor->properties.scale.scaleData[point_idx]);
                            right_val = quanti_scale(data[right], output_tensor->properties.scale.scaleData[point_idx]);
                        }
                        else if (output_tensor->properties.quantiType == NONE)
                        {
                            left_val = data[left];
                            right_val = data[right];
                        }
                        else
                        {
                            return -1;
                        }

                        diff = right_val - left_val;
                    }
                    if (diff > 0)
                    {
                        diff = 0.25;
                    }
                    else if (diff < 0)
                    {
                        diff = -0.25;
                    }
                    else
                    {
                        diff = 0;
                    }
                    max_index = max_index + (diff * 1.0 + 0.5);

                    // calculate the score of each landmarks
                    auto cal_score = [&max_value](float score) -> float
                    {
                        float new_score = 0.0;
                        float val = max_value / 2.0;
                        if (val > 1.0)
                        {
                            val = 1.0;
                        }
                        if (score > 0)
                        {
                            new_score = std::min(score, val);
                        }
                        else
                        {
                            new_score = val;
                        }
                        return new_score;
                    };

                    // save face landmarks to result
                    if (axis == 0)
                    {
                        // x coordinate
                        face_landmarks_det_result->values.at(b).at(point_idx).x = max_index * roi_width / vector_size + roi.left;
                        face_landmarks_det_result->values.at(b).at(point_idx).score = cal_score(face_landmarks_det_result->values.at(b).at(point_idx).score);
                    }
                    else
                    {
                        //  y coordinate
                        face_landmarks_det_result->values.at(b).at(point_idx).y = max_index * roi_height / vector_size + roi.top;
                        face_landmarks_det_result->values.at(b).at(point_idx).score = cal_score(face_landmarks_det_result->values.at(b).at(point_idx).score);
                    }
                }
            }
        }
    }

    return 0;
}
