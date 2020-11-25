//#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include "meshing_occupancy_shared.h"

static const float vertice_offsets[8][3] {
        {-0.5,-0.5,-0.5},
        {-0.5,-0.5,0.5},
        {-0.5,0.5,-0.5},
        {-0.5,0.5,0.5},
        {0.5,-0.5,-0.5},
        {0.5,-0.5,0.5},
        {0.5,0.5,-0.5},
        {0.5,0.5,0.5}
};

static const int32_t index_offset[12][3]={
{0,1,3},{0,3,2},
{4,0,2},{4,2,6},
{5,4,6},{5,6,7},
{1,5,7},{1,7,3},
{2,3,7},{2,7,6},
{1,0,4},{1,4,5}
};




std::vector<at::Tensor> occupancy_meshing_forward(
        torch::Tensor volume,
        torch::optional<torch::Tensor> label = torch::nullopt,
        float threshold = 0.5,
        float voxel_scale = 0.8,
        bool use_cuda = false
        ) {
    assert(volume.dim() == 3);
    auto X = volume.size(0);
    auto Y = volume.size(1);
    auto Z = volume.size(2);
    auto size = volume.numel();

    //std::cout << "volume.sum(): " << volume.sum() << std::endl;
    if(use_cuda){
        assert(volume.is_cuda());
        if(label != torch::nullopt) {
            assert(label->sizes() == volume.sizes());
            assert(label->is_cuda());
        }
    } else {
        assert(!volume.is_cuda());
        if(label != torch::nullopt) {
            assert(label->sizes() == volume.sizes());
            assert(!label->is_cuda());
        }
    }



    auto device = volume.device();
    int32_t *in_label_data = nullptr;
    float *volume_data = volume.data_ptr<float>();// use_cuda ? volume.cuda().data_ptr<float>() : volume.cpu().data_ptr<float>();

    int counter = 0;
    if (use_cuda)
        counter = classifyVoxel(size, volume_data, threshold);
    else
        for (int64_t i = 0; i < size; ++i) if (volume_data[i] > threshold) counter++;
    //printf("Counter: %d\n", counter);


    auto output_vertices = torch::zeros({counter * 8, 3},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(
                                                false));
    auto output_faces = torch::zeros({counter * 12, 3},
                                     torch::TensorOptions().dtype(torch::kInt32).device(device).requires_grad(false));
    torch::optional<torch::Tensor> output_label = torch::nullopt;

    if (label != torch::nullopt) {
        in_label_data = label->data_ptr<int32_t>();
        output_label = torch::zeros({counter * 8, 1},
                                    torch::TensorOptions().dtype(torch::kInt32).device(device).requires_grad(false));
    }

    float *vertices_data = output_vertices.data_ptr<float>();
    int32_t *faces_data = output_faces.data_ptr<int32_t>();
    int32_t *label_data = label == torch::nullopt ? nullptr : output_label->data_ptr<int32_t>();

    if (use_cuda) {
        process(size, X, Y, Z,
                volume_data,
                in_label_data,
                vertices_data,
                faces_data,
                label_data,
                threshold, voxel_scale);
    } else {
        counter = 0;
        auto face_counter = 0;
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y; ++y) {
                for (int z = 0; z < Z; ++z) {
                    int idx = (x * Y + y) * Z + z;
                    if (volume_data[idx] > threshold) {
                        for (auto off : index_offset) {
                            faces_data[face_counter * 3 + 0] = off[0] + counter;
                            faces_data[face_counter * 3 + 1] = off[1] + counter;
                            faces_data[face_counter * 3 + 2] = off[2] + counter;
                            face_counter++;
                        }

                        for (auto vertice_offset : vertice_offsets) {
                            vertices_data[counter * 3 + 0] = x + vertice_offset[0] * voxel_scale;
                            vertices_data[counter * 3 + 1] = y + vertice_offset[1] * voxel_scale;
                            vertices_data[counter * 3 + 2] = z + vertice_offset[2] * voxel_scale;
                            if (label != torch::nullopt)
                                label_data[counter] = in_label_data[idx];
                            counter++;
                        }

                    }
                }
            }
        }
    }
    if (label != torch::nullopt)
        return {output_vertices, output_faces, output_label.value()};
    else
        return {output_vertices, output_faces};
}


at::Tensor label_to_color(
        torch::Tensor label,
        torch::Tensor color,
        bool use_cuda) {
//    printf("label start\n");
//    std::cout << label.dim() << std::endl;
//    std::cout << color.dim() << std::endl;
    if(use_cuda){
        assert(label.is_cuda());
        assert(color.is_cuda());
    } else {
        assert(!label.is_cuda());
        assert(!color.is_cuda());
    }
//    printf("label start2\n");
    assert(label.device() == color.device());
    assert(label.scalar_type() == torch::ScalarType::Int);
    assert(color.scalar_type() == torch::ScalarType::Long);

    assert(color.dim() == 2);
    assert(color.size(1) == 3);
    assert(label.numel() == label.size(0) *label.size(1)*label.size(2)*label.size(3) );

    auto device = label.device();
    torch::Tensor output = torch::zeros({label.numel(), 3},
                               torch::TensorOptions().dtype(torch::kLong).device(device).requires_grad(
                                       false));

    auto color_data = color.data_ptr<int64_t>();
    auto label_data = label.data_ptr<int>();
    auto output_data = output.data_ptr<int64_t>();

    if(use_cuda) {
//        printf("Label:CUDA!\n");
        label2color(label.numel(), label_data,color_data, output_data);
    } else {
//        printf("Label:CPU!\n");
        for(size_t i=0; i < label.numel(); ++i) {
            output_data[i * 3 + 0] = color_data[label_data[i] * 3 + 0];
            output_data[i * 3 + 1] = color_data[label_data[i] * 3 + 1];
            output_data[i * 3 + 2] = color_data[label_data[i] * 3 + 2];
        }
    }
    return output;
}

template<class T> void write_binary_shortcut(std::fstream &file, T data){
    file.write(reinterpret_cast<char*>(&data),sizeof(T));
}

#include <fstream>
void save_to_mesh (const std::string &path,
                    torch::Tensor vertices,
                   torch::optional<torch::Tensor> triangles = torch::nullopt,
                   torch::optional<torch::Tensor> colors = torch::nullopt,
                   torch::optional<torch::Tensor> normals = torch::nullopt){
    bool write_binary = true;
    assert(vertices.dim() == 2);
    assert(vertices.size(1) == 3);
    assert(vertices.scalar_type() == torch::ScalarType::Float);

    if(triangles != torch::nullopt){
        assert(triangles->dim()==2);
        assert(triangles->size(1)==3);
        assert(triangles->scalar_type() == torch::ScalarType::Int);
    }

    if(colors != torch::nullopt){
        assert(colors->dim()==2);
        assert(colors->size(1)==3);
        assert(colors->scalar_type() == torch::ScalarType::Long);
    }
    if(normals != torch::nullopt){
        assert(normals->dim()==2);
        assert(normals->size(1)==3);
        assert(normals->scalar_type() == torch::ScalarType::Float);
    }

    std::fstream fout(path, std::ios::out);
    assert(fout.is_open());

    fout << "ply" << std::endl;
    if(!write_binary)
        fout << "format ascii 1.0" << std::endl;
    else
        fout << "format binary_little_endian 1.0" << std::endl;
    fout << "element vertex " << vertices.size(0) << std::endl;
    fout << "property float x" << std::endl;
    fout << "property float y" << std::endl;
    fout << "property float z" << std::endl;
    if(colors != torch::nullopt) {
        fout << "property uchar red" << std::endl;
        fout << "property uchar green" << std::endl;
        fout << "property uchar blue" << std::endl;
        //fout << "property uchar alpha" << std::endl;
    }
    if(normals != torch::nullopt) {
        fout << "property float nx" << std::endl;
        fout << "property float ny" << std::endl;
        fout << "property float nz" << std::endl;
        fout << "property float curvature" << std::endl;
    }
    if(triangles != torch::nullopt)
        fout << "element face " << triangles->size(0) << std::endl;
    else
        fout << "element face " << std::floor(vertices.size(0)/3) << std::endl;
    fout << "property list uchar int vertex_indices" << std::endl;
    fout << "end_header" << std::endl;

    if(write_binary) {
        fout.close();
        fout.open(path, std::ios::out | std::ios::binary | std::ios::app);
    }
    auto vertice_data = vertices.data_ptr<float>();
    int64_t *color_data=colors == torch::nullopt? nullptr : colors->data_ptr<int64_t>();
    float *normal_data=normals == torch::nullopt?nullptr:normals->data_ptr<float>();
    int *triangle_data=triangles == torch::nullopt?nullptr:triangles->data_ptr<int>();

    for (size_t i=0; i < vertices.size(0); ++i){
        if(!write_binary) {
            fout << vertice_data[i * 3 + 0] << " " << vertice_data[i * 3 + 1] << " " << vertice_data[i * 3 + 2];
            if (color_data)
                fout << " " << color_data[i * 3 + 0] << " " << color_data[i * 3 + 1] << " " << color_data[i * 3 + 2];
            if (normal_data)
                fout << " " << normal_data[i * 3 + 0] << " " << normal_data[i * 3 + 1] << " " << normal_data[i * 3 + 2]
                     << "1";
            fout << "\n";
        } else {
            write_binary_shortcut(fout, vertice_data[i*3+0]);
            write_binary_shortcut(fout, vertice_data[i*3+1]);
            write_binary_shortcut(fout, vertice_data[i*3+2]);
            if(color_data){
                unsigned char r = color_data[i*3+0];
                unsigned char g = color_data[i*3+1];
                unsigned char b = color_data[i*3+2];
                write_binary_shortcut(fout, static_cast<unsigned char>(r));
                write_binary_shortcut(fout, static_cast<unsigned char>(g));
                write_binary_shortcut(fout, static_cast<unsigned char>(b));
            }
            if(normal_data){
                write_binary_shortcut(fout, normal_data[i*3+0]);
                write_binary_shortcut(fout, normal_data[i*3+1]);
                write_binary_shortcut(fout, normal_data[i*3+2]);
            }
        }
    }
    if (triangle_data) {
        if(!write_binary) {
            for (size_t i = 0; i < triangles->size(0); ++i) {
                fout << "3 " << triangle_data[3 * i + 0] << " " << triangle_data[3 * i + 1] << " "
                     << triangle_data[3 * i + 2] << "\n";
            }
        } else {
            for (size_t i = 0; i < triangles->size(0); ++i) {
                write_binary_shortcut(fout, static_cast<unsigned char>(3));
                write_binary_shortcut(fout, static_cast<int>(triangle_data[3 * i + 0]));
                write_binary_shortcut(fout, static_cast<int>(triangle_data[3 * i + 1]));
                write_binary_shortcut(fout, static_cast<int>(triangle_data[3 * i + 2]));
            }
        }
    } else {
        if(!write_binary) {
            for (size_t i = 0; i < vertices.size(0) / 3; ++i) {
                fout << "3 " << 3 * i << " " << 3 * i + 1 << " " << 3 * i + 2 << "\n";
            }
        }else {
            for (size_t i = 0; i < triangles->size(0); ++i) {
                write_binary_shortcut(fout, static_cast<unsigned char>(3));
                write_binary_shortcut(fout, static_cast<int>(3*i));
                write_binary_shortcut(fout, static_cast<int>(3*i+1));
                write_binary_shortcut(fout, static_cast<int>(3*i+2));
            }
        }
    }
    fout.close();
};

#ifndef CPP_DEBUG

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &occupancy_meshing_forward, "occupancy meshing forward",
            py::arg("volume"),
            py::arg("label") = nullptr,
            py::arg("threshold") = 0.5,
            py::arg("voxel_scale") = 0.8,
            py::arg("use_cuda") = false
            );

    m.def("label2color", &label_to_color, "convert label to color",
            py::arg("label"),
            py::arg("color"),
            py::arg("use_cuda") = false
            );
    m.def("write_ply", &save_to_mesh, "save to mesh",
            py::arg("path"),
            py::arg("vertices"),
            py::arg("triangles") = nullptr,
            py::arg("colors") = nullptr,
            py::arg("normals") = nullptr
            );
}
#else

int main(int argc, char** argv){
//    auto a = torch::rand({3,3,3});
//    std::cout << a.device() << std::endl;
//    a = a.to(torch::DeviceType::CUDA);
//
//    auto b = occupancy_meshing_forward(a.cuda(),torch::nullopt, 0.5, 0.9, false);
//    auto c = occupancy_meshing_forward(a.cuda(),torch::nullopt, 0.5, 0.9, true);
//    for(auto &cc: c) cc.to(torch::DeviceType::CPU);
//
//    std::cout << b[0].sizes() << std::endl;
//    std::cout << c[0].sizes() << std::endl;
//
//    printf("values\n");
//    for(size_t i=0;i<b[0].size(0); ++i)
//        std::cout << i << "\t" << b[0][i] << "\n\t" << c[0][i] << "\n\n";

    auto label = torch::randint(0,14,{1,3,3,3}).toType(torch::ScalarType::Int);
    auto color = torch::randint(0,255,{15,3}).toType(torch::ScalarType::Long);
    std::cout << label << std::endl;
    std::cout << color << std::endl;
    auto output = label_to_color(label,color, false);
    std::cout << output[0] << std::endl;

    std::cout << label.device() << std::endl;
    std::cout << label.cuda().device() << std::endl;
    auto output_cuda = label_to_color(label.cuda(),color.cuda(), true);
    std::cout << output_cuda.cpu() << std::endl;

    return 0;
}

#endif
