#include <iostream>
//#include "utils/tum_utils.h"

#include <pangolin/pangolin.h>
#include <pangolin/display.h>
#include <pangolin/glcuda.h>
#include<vector>
#include<Eigen/Eigen>
#include<TooN/se3.h>
#include<TooN/TooN.h>
#include<TooN/Cholesky.h>
#include<string>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
//#include <iu/iucore.h>
//#include <iu/iuio.h>

#include <cvd/gl_helpers.h>
#include <cvd/image.h>
#include <cvd/image_io.h>


//#include "utils/map_object_label2training_label.h"
//#include "utils/povray_utils.h"
#include "src/tinyobjloader/tiny_obj_loader.h"

using namespace pangolin;

static void PrintInfo(const std::vector<tinyobj::shape_t>& shapes, const std::vector<tinyobj::material_t>& materials)
{
  std::cout << "# of shapes    : " << shapes.size() << std::endl;
  std::cout << "# of materials : " << materials.size() << std::endl;

  for (size_t i = 0; i < shapes.size(); i++) {
    printf("shape[%ld].name = %s\n", i, shapes[i].name.c_str());
//    printf("Size of shape[%ld].indices: %ld\n", i, shapes[i].mesh.indices.size());
//    printf("Size of shape[%ld].material_ids: %ld\n", i, shapes[i].mesh.material_ids.size());
    assert((shapes[i].mesh.indices.size() % 3) == 0);
//    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
//      printf("  idx[%ld] = %d, %d, %d. mat_id = %d\n", f, shapes[i].mesh.indices[3*f+0], shapes[i].mesh.indices[3*f+1], shapes[i].mesh.indices[3*f+2], shapes[i].mesh.material_ids[f]);
//    }

//    printf("shape[%ld].vertices: %ld\n", i, shapes[i].mesh.positions.size());
//    assert((shapes[i].mesh.positions.size() % 3) == 0);
//    for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {
//      printf("  v[%ld] = (%f, %f, %f)\n", v,
//        shapes[i].mesh.positions[3*v+0],
//        shapes[i].mesh.positions[3*v+1],
//        shapes[i].mesh.positions[3*v+2]);
//    }
  }

  for (size_t i = 0; i < materials.size(); i++) {
    printf("material[%ld].name = %s\n", i, materials[i].name.c_str());
    printf("  material.Ka = (%f, %f ,%f)\n", materials[i].ambient[0], materials[i].ambient[1], materials[i].ambient[2]);
    printf("  material.Kd = (%f, %f ,%f)\n", materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
    printf("  material.Ks = (%f, %f ,%f)\n", materials[i].specular[0], materials[i].specular[1], materials[i].specular[2]);
    printf("  material.Tr = (%f, %f ,%f)\n", materials[i].transmittance[0], materials[i].transmittance[1], materials[i].transmittance[2]);
    printf("  material.Ke = (%f, %f ,%f)\n", materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
    printf("  material.Ns = %f\n", materials[i].shininess);
    printf("  material.Ni = %f\n", materials[i].ior);
    printf("  material.dissolve = %f\n", materials[i].dissolve);
    printf("  material.illum = %d\n", materials[i].illum);
    printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
    printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
    printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
    printf("  material.map_Ns = %s\n", materials[i].normal_texname.c_str());
    std::map<std::string, std::string>::const_iterator it(materials[i].unknown_parameter.begin());
    std::map<std::string, std::string>::const_iterator itEnd(materials[i].unknown_parameter.end());
    for (; it != itEnd; it++) {
      printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
    }
    printf("\n");
  }
}

bool TestLoadObj(
  const char* filename,
//  const char* basepath = NULL,
  std::vector<tinyobj::shape_t>& shapes,
  std::vector<tinyobj::material_t>& materials)
{
  std::cout << "Loading " << filename << std::endl;

//  std::vector<tinyobj::shape_t> shapes;
//  std::vector<tinyobj::material_t> materials;
  std::string err = tinyobj::LoadObj(shapes, materials, filename, NULL);

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  PrintInfo(shapes, materials);

  return true;
}


//#define RADPERDEG 0.0174533

void change_basis(TooN::SE3<>& T_wc_ref,
                  TooN::Cholesky<4>& Tchangebasis,
                  TooN::Matrix<4>&T)
{
    TooN::Matrix<4>T4x4 = T/*.T()*/ * T_wc_ref /** T  */;

    TooN::Matrix<3>R_slice = TooN::Data(T4x4(0,0),T4x4(0,1),T4x4(0,2),
                                        T4x4(1,0),T4x4(1,1),T4x4(1,2),
                                        T4x4(2,0),T4x4(2,1),T4x4(2,2));


    TooN::Vector<3>t_slice = TooN::makeVector(T4x4(0,3),T4x4(1,3),T4x4(2,3));

    T_wc_ref = TooN::SE3<>(TooN::SO3<>(R_slice),t_slice);

}

int main(int argc, char *argv[])
{

    if ( argc < 2)
    {
        std::cerr<<"Usage: ./binary_name obj_file_path " << std::endl;
        std::cerr<<"example: ./tinyobj_read_model_and_display ../data/room_89_simple.obj " << std::endl;
        exit(1);
    }

    std::string obj_basename(argv[1]);
    std::size_t find_dot = obj_basename.find(".obj");
    std::size_t find_slash = obj_basename.find_last_of('/');

    std::cout<<"find_dot = " << find_dot << std::endl;
    std::cout<<"find_slash = " << find_slash << std::endl;

    obj_basename = obj_basename.substr(find_slash+1,find_dot-find_slash-1);

    std::cout<<"obj_basename = " << obj_basename << std::endl;


    std::string data_dir = "../data/" + obj_basename + "_data";

    /// Simple OBJ reader - main code begins after this..

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    TestLoadObj(argv[1],shapes,materials);

    std::vector<float*>shape_vertices(shapes.size(),NULL);

    /// Reading the obj mesh
    ///
    ///
    float maxx = -10000, minx = 10000;
    float maxy = -10000, miny = 10000;
    float maxz = -10000, minz = 10000;

    for(int i = 0; i < shape_vertices.size(); i++)
    {
        int num_vertices = shapes[i].mesh.positions.size()/3;
        int num_faces    = shapes[i].mesh.indices.size() / 3;

        shape_vertices[i] = new float[num_faces*3*3];

        int count=0;

        for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++)
        {
            int v1_idx = shapes[i].mesh.indices[3*f+0];
            int v2_idx = shapes[i].mesh.indices[3*f+1];
            int v3_idx = shapes[i].mesh.indices[3*f+2];

            int max_index = std::max(std::max(v1_idx,v2_idx),v3_idx);

            if ( max_index > num_vertices )
            {
                std::cerr<<"max_vertex_index exceeds the number of vertices, something fishy!" << std::endl;
                return 1;
            }

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v1_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v1_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v1_idx+2];
            if (shape_vertices[i][count+0] > maxx) maxx = shape_vertices[i][count+0];
            if (shape_vertices[i][count+0] < minx) minx = shape_vertices[i][count+0];
            if (shape_vertices[i][count+1] > maxy) maxy = shape_vertices[i][count+1];
            if (shape_vertices[i][count+1] < miny) miny = shape_vertices[i][count+1];
            if (shape_vertices[i][count+2] > maxz) maxz = shape_vertices[i][count+2];
            if (shape_vertices[i][count+2] < minz) minz = shape_vertices[i][count+2];
            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v2_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v2_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v2_idx+2];
            if (shape_vertices[i][count+0] > maxx) maxx = shape_vertices[i][count+0];
            if (shape_vertices[i][count+0] < minx) minx = shape_vertices[i][count+0];
            if (shape_vertices[i][count+1] > maxy) maxy = shape_vertices[i][count+1];
            if (shape_vertices[i][count+1] < miny) miny = shape_vertices[i][count+1];
            if (shape_vertices[i][count+2] > maxz) maxz = shape_vertices[i][count+2];
            if (shape_vertices[i][count+2] < minz) minz = shape_vertices[i][count+2];
            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v3_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v3_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v3_idx+2];
            if (shape_vertices[i][count+0] > maxx) maxx = shape_vertices[i][count+0];
            if (shape_vertices[i][count+0] < minx) minx = shape_vertices[i][count+0];
            if (shape_vertices[i][count+1] > maxy) maxy = shape_vertices[i][count+1];
            if (shape_vertices[i][count+1] < miny) miny = shape_vertices[i][count+1];
            if (shape_vertices[i][count+2] > maxz) maxz = shape_vertices[i][count+2];
            if (shape_vertices[i][count+2] < minz) minz = shape_vertices[i][count+2];
            count+=3;

        }
    }

    std::cout<<"Min Max values"<<std::endl;
    std::cout<<minx<<" "<<maxx<<std::endl;
    std::cout<<miny<<" "<<maxy<<std::endl;
    std::cout<<minz<<" "<<maxz<<std::endl;

    //float sphere_radius = 0.5 * 1.5*std::max(std::max(maxx-minx,maxy-miny),maxz-minz);
    float sphere_radius = 0.5*1.5*sqrt((maxx-minx)*(maxx-minx)+(maxy-miny)*(maxy-miny)+(maxz-minz)*(maxz-minz));
    std::cout<<"Sphere radius = "<<sphere_radius<<std::endl;

//    TooN::SE3<>T_move;

    std::cout<<"entering the while loop" << std::endl;

    OpenGlMatrix openglSE3Matrix;

    std::vector<TooN::SE3<> >poses2render;
    poses2render.push_back(TooN::SE3<>());

    int render_pose_count = 0;

    int n_levels = 10;
    int l_levels = 10;
    float rx = 0, ry = 0, rz = 0;
    float tx = 0, ty = 0, tz = 0;
//    for(int i = (int)n_levels; i >=(int)((int)n_levels-(int)l_levels) ; i--)
//    {
//        float theta = (i * M_PI) / ( (int)n_levels * 1.0f) ;

//        for(int j = 0; j < 4*i; j++)
//        {
//            float phi = ( (1/2.0f) * j * (float)M_PI) / i ; /// [0, 2 * M_PI]

    for (int i=0; i<=n_levels; ++i)
    {
        float theta = (i * M_PI) / ( (int)n_levels * 1.0f) ;

        for (int j=0; j<=l_levels; ++j)
        {
            float phi = (j * 2*M_PI) / ( (int)l_levels * 1.0f) ;
            float3 point = make_float3(sphere_radius*sin(theta)*cos(phi),
                                       sphere_radius*sin(theta)*sin(phi),
                                       sphere_radius*cos(theta));

            //  sphere_points.push_back(point);

            TooN::Matrix<3>RotMat = TooN::Zeros(3);

            /// http://www.iac.ethz.ch/edu/courses/bachelor/veranstaltungen/environmental_fluid_dynamics/AD3
            /// Page 4!

            TooN::Vector<3>zAxis =  -1.0f* TooN::makeVector(point.x,point.y,point.z)/(float)sphere_radius;
            TooN::Vector<3>xAxis =  -1.0f* TooN::makeVector(-sin(phi),cos(phi),0);
            TooN::Vector<3>yAxis =  zAxis ^ xAxis;

            RotMat.T()[0] = xAxis;
            RotMat.T()[1] = yAxis;
            RotMat.T()[2] = zAxis;

            TooN::SO3<>RMat(RotMat);

            TooN::SE3<>T_ = TooN::SE3<>(RMat,TooN::makeVector(point.x,
                                                              point.y,
                                                              point.z));

            TooN::SE3<>T_rot = TooN::SE3<>(TooN::SO3<>(TooN::makeVector((float)rx,(float)ry,(float)rz)),
                                           TooN::makeVector((float)tx,(float)ty,(float)tz));

            TooN::SE3<>T_plot = T_rot * T_;

            poses2render.push_back(T_plot);
        }

    }


    char trajectory_fileName[200];

//    sprintf(trajectory_fileName,"%s/%s_trajectory_random_poses_SE3_3x4.txt",
//            data_dir.c_str(),
//            obj_basename.c_str());

    sprintf(trajectory_fileName,"../data/poses/%s_bounding_sphere_poses.txt",obj_basename.c_str());

    std::cout<<"trajectory_fileName = " << trajectory_fileName << std::endl;

    std::ofstream ofile;
    ofile.open(trajectory_fileName);

    for(int i = 0 ;i < poses2render.size(); i++)
        ofile << poses2render.at(i) << std::endl;

    ofile.close();

    return 0;
}
