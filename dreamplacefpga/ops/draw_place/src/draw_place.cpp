/**
 * @file   src/draw_place.cpp
 * @author Yibo Lin (DREAMPlace)
 * @date   Jun 2018
 * @brief  Plot placement to an image 
 */
#include <sstream>
#include "utility/src/torch.h"
#include "draw_place/src/draw_place.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief plot placement solution to an image 
/// @param pos cell locations, array of x locations and then y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param pin_offset_x pin offset to its cell origin
/// @param pin_offset_y pin offset to its cell origin
/// @param pin2node_map map pin to cell 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param site_width width of a placement site 
/// @param row_height height of a placement row, same as height of a placement site 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param num_movable_nodes number of movable cells 
/// @param num_filler_nodes number of filler cells 
/// @param filename output image file name 
int draw_place_forward(
        at::Tensor pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor pin_offset_x, 
        at::Tensor pin_offset_y, 
        at::Tensor pin2node_map, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double site_width, 
        double row_height, 
        double bin_size_x, 
        double bin_size_y, 
        int num_movable_nodes, 
        int num_filler_nodes, 
        const std::string& filename
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_nodes = pos.numel()/2; 

    // Call the kernel launcher
    int ret = 0; 
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "drawPlaceLauncher", [&] {
            ret = drawPlaceLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+num_nodes, 
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int), 
                    num_nodes, 
                    num_movable_nodes, 
                    num_filler_nodes, 
                    pin2node_map.numel(), 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    bin_size_x, bin_size_y, 
                    filename
                    );
            });

    return ret; 
}

int draw_place_fpga(
        at::Tensor pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor pin_offset_x, 
        at::Tensor pin_offset_y, 
        at::Tensor pin2node_map, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double bin_size_x, 
        double bin_size_y, 
        int num_physical_nodes, 
        int num_filler_nodes, 
        at::Tensor node2fence_region_map,
        at::Tensor is_cc_node,
        int ffIdx,
        int lutIdx,
        int addIdx,
        int bramIdx,
        int m9kIdx,
        int m144kIdx,
        int dspIdx,
        int ioIdx,
        int pllIdx,
        const std::string& filename
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_nodes = pos.numel()/2; 

    // Call the kernel launcher
    int ret = 0; 
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "drawPlaceLauncherFPGA", [&] {
            ret = drawPlaceLauncherFPGA<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+num_nodes, 
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int), 
                    num_nodes, 
                    num_physical_nodes, 
                    num_filler_nodes, 
                    pin2node_map.numel(), 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    DREAMPLACE_TENSOR_DATA_PTR(node2fence_region_map, int),
                    DREAMPLACE_TENSOR_DATA_PTR(is_cc_node, int),
                    ffIdx, lutIdx, addIdx,
                    bramIdx, m9kIdx, m144kIdx,
                    dspIdx,
                    ioIdx, pllIdx,
                    filename
                    );
            });

    return ret; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::draw_place_forward, "Draw place forward");
  m.def("fpga", &DREAMPLACE_NAMESPACE::draw_place_fpga, "Draw place FPGA");
}
