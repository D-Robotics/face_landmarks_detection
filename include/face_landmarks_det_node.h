#ifndef FACE_LANDMARKS_DET_NODE_H
#define FACE_LANDMARKS_DET_NODE_H

#include "rclcpp/rclcpp.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "ai_msgs/msg/perception_targets.hpp"
#include "img_convert_utils.h"
#include "face_landmarks_det_output_parser.h"

using rclcpp::NodeOptions;

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::DnnNodePara;

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::ModelRoiInferTask;
using hobot::dnn_node::ModelTaskType;
using hobot::dnn_node::NV12PyramidInput;

using ai_msgs::msg::PerceptionTargets;

/**
 * @brief image used for feedback
 */
struct FeedbackImgInfo
{
    std::string image;
    int img_w;
    int img_h;
    std::vector<std::vector<int32_t>> rois;
};

/**
 * @brief face landmarks det output tensor
 */
struct FaceLandmarksDetOutput : public DnnNodeOutput
{
    // svae image timestamp
    std::shared_ptr<std_msgs::msg::Header> image_msg_header = nullptr;

    // roi that meets the constraints of the resizer model
    std::shared_ptr<std::vector<hbDNNRoi>> valid_rois;

    // roi index
    std::map<size_t, size_t> valid_roi_idx;

    // image used for model inference: used for local rendering
    std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;

    // pub ai_msg
    ai_msgs::msg::PerceptionTargets::UniquePtr ai_msg;

    // recording time-consuming
    ai_msgs::msg::Perf perf_preprocess;
};

/**
 * @brief face landmarks detection
 */
class FaceLandmarksDetNode : public DnnNode
{
public:
    explicit FaceLandmarksDetNode(const std::string &node_name = "face_landmarks_det_node", const NodeOptions &options = NodeOptions());
    ~FaceLandmarksDetNode() override;

protected:
    /**
     * @brief init DnnNode param
     */
    int SetNodePara() override;

    /**
     * @brief post process
     */
    int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

private:
    // =================================================================================================================================
    /**
     * @brief using local images for inference
     */
    int Feedback();

    /**
     * @brief do model inference -- offline
     */
    int Predict(std::vector<std::shared_ptr<DNNInput>> &inputs, const std::shared_ptr<std::vector<hbDNNRoi>> rois, std::shared_ptr<DnnNodeOutput> dnn_output);

    /**
     * @brief render result to image and save it
     */
    int Render(const std::shared_ptr<NV12PyramidInput> &pyramid, std::string result_image, std::shared_ptr<std::vector<hbDNNRoi>> &valid_rois, std::shared_ptr<FaceLandmarksDetResult> &face_landmarks_det_result);

    // =================================================================================================================================
    // image source used for inference, 0: subscribed image msg; 1: local nv12 format image
    int feed_type_ = 0;
    // image used for feedback
    std::string feed_image_path_ = "./config/image.png";
    // image used for feedback
    FeedbackImgInfo fb_img_info_;

    // model bin file path
    std::string model_file_name_ = "./config/faceLandmark106pts.hbm";
    // model name
    std::string model_name_ = "faceLandmark106pts";

    // model input & output info
    int model_input_width_ = -1;
    int model_input_height_ = -1;
    int32_t model_output_count_ = 1;

    // mode task type
    ModelTaskType model_task_type_ = ModelTaskType::ModelRoiInferType;

    // 0: asynchronous inference, 1: synchronous inference
    int is_sync_mode_ = 0;

    // subscribe image using shared mem communication method
    int is_shared_mem_sub_ = 1;

    // save the rendered image
    int dump_render_img_ = 0;
    // rendered image count
    int render_count_ = 0;

    // ai_msg pub topic
    std::string ai_msg_pub_topic_name_ = "/face_landmarks_detection";
    // ai_msg pub
    rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr ai_msg_publisher_ = nullptr;

    // ai_msg sub topic
    std::string ai_msg_sub_topic_name_ = "/hobot_mono2d_body_detection";
    // ai_msg sub
    rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr ai_msg_subscription_ = nullptr;

    // image sub topic
    // Supports sub to the original image | compressed image "/image_raw/compressed" | sensor_msgs::msg::CompressedImage
    std::string ros_img_topic_name_ = "/image_raw";
    // image sub
    rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr ros_img_subscription_ = nullptr;

    // use to process ai msg
    // std::shared_ptr<AiMsgManage> ai_msg_manage_ = nullptr;

    // predict task
    std::shared_ptr<std::thread> predict_task_ = nullptr;

    // Convert the subscribed image data into pym and cache it in the queue
    // using CacheImgType = std::pair<std::shared_ptr<FaceAgeDetOutput>, std::shared_ptr<NV12PyramidInput>>;
    // std::queue<CacheImgType> cache_img_;
    // size_t cache_len_limit_ = 8;

    // Perform inference in threads to avoid blocking IO channels and causing AI msg message loss
    std::mutex mtx_img_;
    std::condition_variable cv_img_;
};

#endif