#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <TooN/TooN.h>
#include <TooN/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/display.h>
#include <pangolin/glcuda.h>
#include <cvd/image_io.h>

#include <cvd/gl_helpers.h>

#include "src/utils/povray_utils.h"
#include "src/tinyobjloader/tiny_obj_loader.h"
#include "utils/map_object_label2training_label.h"

using namespace pangolin;
using namespace std;

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


#define RADPERDEG 0.0174533

void Arrow(GLdouble x1,GLdouble y1,GLdouble z1,GLdouble x2,GLdouble y2,GLdouble z2,GLdouble D)
{
  double x=x2-x1;
  double y=y2-y1;
  double z=z2-z1;
  double L=sqrt(x*x+y*y+z*z);

    GLUquadricObj *quadObj;

    glPushMatrix ();

      glTranslated(x1,y1,z1);

      if((x!=0.)||(y!=0.)) {
        glRotated(atan2(y,x)/RADPERDEG,0.,0.,1.);
        glRotated(atan2(sqrt(x*x+y*y),z)/RADPERDEG,0.,1.,0.);
      } else if (z<0){
        glRotated(180,1.,0.,0.);
      }

      glTranslatef(0,0,L-4*D);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluCylinder(quadObj, 2*D, 0.0, 4*D, 32, 1);
      gluDeleteQuadric(quadObj);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluDisk(quadObj, 0.0, 2*D, 32, 1);
      gluDeleteQuadric(quadObj);

      glTranslatef(0,0,-L+4*D);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluCylinder(quadObj, D, D, L-4*D, 32, 1);
      gluDeleteQuadric(quadObj);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluDisk(quadObj, 0.0, D, 32, 1);
      gluDeleteQuadric(quadObj);

    glPopMatrix ();

}
void drawAxes(GLdouble length)
{
    glPushMatrix();
    glColor3f(1.0,0,0);
    glTranslatef(-length,0,0);
    Arrow(0,0,0, 2*length,0,0, 0.1);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.0,1.0,0);
    glTranslatef(0,-length,0);
    Arrow(0,0,0, 0,2*length,0, 0.1);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.0,0.0,1.0);
    glTranslatef(0,0,-length);
    Arrow(0,0,0, 0,0,2*length, 0.1);
    glPopMatrix();
}


void change_basis(TooN::SE3<>& T_wc_ref,
                  TooN::Matrix<4>&T)
{
    TooN::Matrix<4>T4x4 = T.T() * T_wc_ref * T  ;

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
        std::cerr<<"example: ./render_rgb_depth ../data/room_89_simple.obj" << std::endl;
        exit(1);
    }

    std::string obj_basename(argv[1]);
    std::size_t find_dot   = obj_basename.find(".obj");
    std::size_t find_slash = obj_basename.find_last_of('/');

    std::cout<<"  find_dot = " << find_dot << std::endl;
    std::cout<<"find_slash = " << find_slash << std::endl;

    obj_basename = obj_basename.substr(find_slash+1,find_dot-find_slash-1);

    std::string data_dir = "../data/";// + obj_basename + "_data";

    char trajectory_fileName[300];

    sprintf(trajectory_fileName,"../data/poses/%s_bounding_sphere_poses.txt",
            obj_basename.c_str());

    std::cout<<"Poses filename "<<trajectory_fileName<<std::endl;

    /// Simple OBJ reader - main code begins after this..

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    TestLoadObj(argv[1],shapes,materials);

    std::vector<float*>shape_vertices(shapes.size(),NULL);

    /// Reading the obj mesh
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

            int max_index = max(max(v1_idx,v2_idx),v3_idx);

            if ( max_index > num_vertices )
            {
                std::cerr<<"max_vertex_index exceeds the number of vertices, something fishy!" << std::endl;
                return 1;
            }

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v1_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v1_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v1_idx+2];

            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v2_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v2_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v2_idx+2];

            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v3_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v3_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v3_idx+2];

            count+=3;

        }
    }

    /// Scale 1 means 640x480 images
    /// Scale 2 means 320x240 images

    int scale              = 1;

    int width              = 640/scale;
    int height             = 480/scale;



    int w_width  = 640;
    int w_height = 480;
    const int UI_WIDTH = 150;

    pangolin::CreateGlutWindowAndBind("GUISandbox",w_width+150,w_height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glewInit();

    /// Create a Panel
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
            .SetBounds(1.0, 0.0, 0, pangolin::Attach::Pix(150));


    /// This is the one I used for depth rendering....
    pangolin::OpenGlRenderState s_cam(
      ProjectionMatrixRDF_BottomLeft(640,480,420.0,420.0,320,240,0.1,1000),
      ModelViewLookAt(3,3,3, 0,0,0,AxisNegZ)
    );

    TooN::Matrix<4>T = TooN::Data(1,0,0,0,
                                  0,1,0,0,
                                  0,0,1,0,
                                  0,0,0,1);


    //    ProjectionMatrixRDF_BottomLeft(640,480,420.0,420.0,320,240,0.1,1000),
        float u0 = 320.0;
        float v0 = 240.0;
        float fx = 420.0;
        float fy = -420.0;



    /// Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::Display("cam")
      .SetBounds(0.0, 1, Attach::Pix(UI_WIDTH), 1, -640.0f/480.0f)
      .SetHandler(new Handler3D(s_cam));

    //std::cout<<"entering the while loop" << std::endl;

    OpenGlMatrix openglSE3Matrix;

    TooN::Matrix<17,3>colours = TooN::Data(0,0,1.0,
                                           0.9137,0.3490,0.1882,
                                           0,0.8549,0,
                                           0.5843,0,0.9412,
                                           0.8706,0.9451,0.0941,
                                           1.0000,0.8078,0.8078,
                                           0,0.8784,0.8980,
                                           0.4157,0.5333,0.8000,
                                           0.4588,0.1137,0.1608,
                                           0.9412,0.1333,0.9216,
                                           0,0.6549,0.6118,
                                           0.9765,0.5451,0,
                                           0.8824,0.8980,0.7608,
                                           1.0000,0,0,
                                           0.8118,0.7176,0.2706,
                                           0.7922,0.5804,0.5804,
                                           0.4902,0.4824,0.4784);

    std::map<int,int>colour2indexMap;

    ///SynthCam3D
    colour2indexMap[  0 +   0*256 + 255*256*256] = 1;
    colour2indexMap[233 +  89*256 +  48*256*256] = 2;
    colour2indexMap[  0 + 218*256 +   0*256*256] = 3;
    colour2indexMap[149 +   0*256 + 240*256*256] = 4;
    colour2indexMap[222 + 241*256 +  24*256*256] = 5;
    colour2indexMap[255 + 206*256 + 206*256*256] = 6; // FURNITURE
    colour2indexMap[  0 + 224*256 + 229*256*256] = 7;
    colour2indexMap[106 + 136*256 + 204*256*256] = 8; // SHELF
    colour2indexMap[117 +  29*256 +  41*256*256] = 9;
    colour2indexMap[240 +  34*256 + 235*256*256] = 10;
    colour2indexMap[  0 + 167*256 + 156*256*256] = 11;
//    colour2indexMap[249 + 139*256 +   0*256*256] = 12; //DOOR
    colour2indexMap[225 + 229*256 + 194*256*256] = 13-1;
    colour2indexMap[255 +   0*256 +   0*256*256] = 14-1;
    colour2indexMap[207 + 183*256 +  69*256*256] = 15-1;
    colour2indexMap[202 + 148*256 + 148*256*256] = 16-1;
    colour2indexMap[125 + 123*256 + 122*256*256] = 17-1;
    colour2indexMap[  0 + 0*256   +   0*256*256] = 17;


    std::vector<TooN::SE3<> >poses2render;
    int render_pose_count = 0;

    float depth_arrayf[width*height];

    CVD::Image<u_int16_t>depth_image(CVD::ImageRef(width,height));

    float near =0.1;
    float far = 1000;

    srand (time(NULL));

    CVD::Image<CVD::Rgb<CVD::byte> > img_flipped(CVD::ImageRef(640,480));

    ifstream SE3PoseFile(trajectory_fileName);


    TooN::SE3<>T_wc;

    if (SE3PoseFile.is_open())
    {
        while(1)
        {
            SE3PoseFile >> T_wc;

            if ( SE3PoseFile.eof() )
                break;

            poses2render.push_back(T_wc);

        }

    }
    SE3PoseFile.close();

    std::cout<<"Poses to render "<<poses2render.size()<<std::endl;

    int skip_frame = 1;

    ofstream model_file("3dmodel.obj");

    while(!pangolin::ShouldQuit())
    {
        static Var<int>numposes2plot("ui.numposes2plot",0,0,100);
        static Var<bool>camera_plot("ui.camera_plot",false);
        static Var<float> end_pt("ui.end_pt",0.1,0,10);
        static Var<float> line_width("ui.line_width",2,0,100);
        static Var<bool> do_inverse("ui.do_inverse",false);
        static Var<bool>write_poses("ui.write_poses",true);

        static Var<int>n_levels("ui.n_levels",2,1,20);
        static Var<int>l_levels("ui.l_levels",2,1,20);

        static Var<float>sphere_radius("ui.sphere_radius",0.5,0.1,30);

        static Var<bool>write_poses_file("ui.write_poses_file",false);

        if (camera_plot)
        {
            poses2render.clear();

            //        int n_levels = 10;
            //        int l_levels = 10;
            float rx = 0, ry = 0, rz = 0;
            float tx = 0, ty = 0, tz = 0;
//            for(int i = (int)n_levels; i >=(int)((int)n_levels-(int)l_levels) ; i--)
//            {
//                float theta = (i * M_PI) / ( (int)n_levels * 1.0f) ;

//                for(int j = 0; j < 4*i; j++)
//                {
//                    float phi = ( (1/2.0f) * j* (float)M_PI) / i ; /// [0, 2 * M_PI]

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



            /// DrawCamera expanded at the end of main function

            d_cam.ActivateScissorAndClear(s_cam);

            glEnable(GL_DEPTH_TEST);

            glClear(GL_COLOR_BUFFER_BIT);

            glColor3f(1.0f,1.0f,1.0f);

            for(int i = 0 ; i < shapes.size();i++)
            {
                int training_label = obj_label2training_label(shapes[i].name);

                glColor3f(colours(training_label,0),colours(training_label,1),colours(training_label,2));

                glEnableClientState(GL_VERTEX_ARRAY);

                glVertexPointer(3,GL_FLOAT,0,shape_vertices[i]);
                glDrawArrays(GL_TRIANGLES,0,shapes[i].mesh.indices.size());
                glDisableClientState(GL_VERTEX_ARRAY);

            }

            for(int i =0 ; i < (int)poses2render.size(); i++)
            {
                TooN::SE3<>T_plot = poses2render.at(i);

                povray_utils::DrawCamera(T_plot,
                                         (float)end_pt,
                                         (float)line_width,
                                         do_inverse);
            }





        }

//        else if (write_poses_file)
//        {
//            camera_plot = false;
//            write_poses_file = false;

//            char trajectory_fileName[200];

//        //    sprintf(trajectory_fileName,"%s/%s_trajectory_random_poses_SE3_3x4.txt",
//        //            data_dir.c_str(),
//        //            obj_basename.c_str());

//            sprintf(trajectory_fileName,"../data/poses/%s_bounding_sphere_poses.txt",obj_basename.c_str());

//            std::cout<<"trajectory_fileName = " << trajectory_fileName << std::endl;

//            std::ofstream ofile;
//            ofile.open(trajectory_fileName);

//            for(int i = 0 ;i < poses2render.size(); i++)
//                ofile << poses2render.at(i) << std::endl;

//            ofile.close();

//        }

        else if ( write_poses )
        {
            //write_poses = false;

            camera_plot = false;

            numposes2plot  = render_pose_count;

            if ( numposes2plot >= poses2render.size())
                return 1;

            TooN::SE3<>T_wc = poses2render.at(render_pose_count);

            TooN::SE3<>T_cw = T_wc.inverse();

            TooN::SO3<>Rot = T_cw.get_rotation();

            TooN::Matrix<4>SE3Mat = TooN::Identity(4);

            /// copy rotation
            TooN::Matrix<3>SO3Mat = Rot.get_matrix();            
            SE3Mat.slice(0,0,3,3) = SO3Mat;

            /// copy translation
            TooN::Vector<3>trans = T_cw.get_translation();
            SE3Mat(0,3) = trans[0];
            SE3Mat(1,3) = trans[1];
            SE3Mat(2,3) = trans[2];

            /// Ref: http://www.felixgers.de/teaching/jogl/generalTransfo.html
            /// It should be a transpose - stored in column major
            for(int col = 0; col < 4; col++ )
            {
                for(int row = 0; row < 4; row++)
                {
                    openglSE3Matrix.m[col*4+row] = SE3Mat(row,col);
                }
            }

            /// set the model view matrix to this pose
            s_cam.SetModelViewMatrix(openglSE3Matrix);

            s_cam.Apply();

            numposes2plot = render_pose_count;

            d_cam.ActivateScissorAndClear(s_cam);

            glEnable(GL_DEPTH_TEST);

            glClear(GL_COLOR_BUFFER_BIT);

            glColor3f(1.0f,1.0f,1.0f);

            for(int i = 0 ; i < shapes.size();i++)
            {
                int training_label = obj_label2training_label(shapes[i].name);

                glColor3f(colours(training_label,0),colours(training_label,1),colours(training_label,2));

                glEnableClientState(GL_VERTEX_ARRAY);

                glVertexPointer(3,GL_FLOAT,0,shape_vertices[i]);
                glDrawArrays(GL_TRIANGLES,0,shapes[i].mesh.indices.size());
                glDisableClientState(GL_VERTEX_ARRAY);

            }

            CVD::Image<CVD::Rgb<CVD::byte> > img = CVD::glReadPixels<CVD::Rgb<CVD::byte> >(CVD::ImageRef(640,480),
                                                                                           CVD::ImageRef(150,0));

            /// save the image
#pragma omp parallel for
            for(int yy = 0; yy < height; yy++ )
            {
                for(int xx = 0; xx < width; xx++)
                {
                    img_flipped[CVD::ImageRef(xx,height-1-yy)] = img[CVD::ImageRef(xx,yy)];
                }
            }

            char fileName[300];

            sprintf(fileName,"%s/scene_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);

            CVD::img_save(img_flipped,fileName);

            CVD::Image<CVD::byte>labelImage = CVD::Image<CVD::byte>(CVD::ImageRef(640,480));

            /// save the annotations
#pragma omp parallel for
            for(int yy = 0; yy < height; yy++ )
            {
                for(int xx = 0; xx < width; xx++)
                {
                    CVD::Rgb<CVD::byte> pix = img_flipped[CVD::ImageRef(xx,yy)];

                    int ind = pix.red + 256*pix.green + 256*256*pix.blue;

                    labelImage[CVD::ImageRef(xx,yy)] = colour2indexMap[ind];
                }
            }

            sprintf(fileName,"%s/label_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);
            CVD::img_save(labelImage,fileName);

            glReadPixels(150, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT, depth_arrayf);

            int scale = 500;

            /// convert to real-depth
#pragma omp parallel for
            for(int i = 0; i < width*height; ++i)
            {
                float z_b = depth_arrayf[i];
                float z_n = 2.0f * z_b - 1.0f;
                depth_arrayf[i] = 2.0 * near * far / (far + near - z_n * (far - near));
            }

            /// save the depth image
#pragma omp parallel for
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    int ind = (height-1-y)*width+x;
                    float depth_val = depth_arrayf[ind];

//                    std::cout<<"depth_val " << depth_val << std::endl;

                    if( depth_val * scale < 65535 )
                        depth_image[CVD::ImageRef(x,y)] = (u_int16_t)(depth_val*scale);
                    else
                        depth_image[CVD::ImageRef(x,y)] = 65535 ;
                }
            }

            char depthImageFileName[300];

            {
                sprintf(depthImageFileName,"%s/scenedepth_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);
            }

            std::string depthImageFileName_string(depthImageFileName);

            CVD::img_save(depth_image,depthImageFileName_string);

            char poseFileName[300];

            sprintf(poseFileName,"%s/pose_%07d.txt",data_dir.c_str(),render_pose_count/skip_frame);

            ofstream ofile(poseFileName);

            ofile << T_wc << std::endl;

            ofile.close();

            float rand_r = (float)rand()/RAND_MAX;
            float rand_g = (float)rand()/RAND_MAX;
            float rand_b = (float)rand()/RAND_MAX;

            /// write to obj the backprojected 3d points using the current camera pose
            if( render_pose_count%2 == 0  )
            {

                float max_depth = *std::max_element(depth_arrayf,depth_arrayf+width*height);
                float min_depth = *std::min_element(depth_arrayf,depth_arrayf+width*height);

                std::cout<<"max_depth = " << max_depth << std::endl;
                std::cout<<"min_depth = " << min_depth << std::endl;

                for(int y = 0; y < height; y++)
                {
                    for(int x = 0; x < width; x++)
                    {
                        float depth = (float)depth_image[CVD::ImageRef(x,y)]/scale;

                        if ( depth < 100 && depth > 0 ) // && depth_image[CVD::ImageRef(x,y)] != 65535 )
                        {

                            change_basis(T_wc,T);

                            TooN::Vector<4>p_w = T_wc * TooN::makeVector(depth*(x-u0)/fx,
                                                                         depth*(y-v0)/fy,
                                                                         depth,
                                                                         1.0);

                            model_file << "v " << p_w[0]<<" "<<p_w[1]<<" "<<p_w[2]<<" "<<rand_r<<" "<<rand_g<<" "<<rand_b<<std::endl;

                        }
                    }
                }
            }

        }

        d_panel.Render();
        pangolin::FinishGlutFrame();

        render_pose_count+=skip_frame;

    }

    model_file.close();

}


//float projectionMatrix[16];
//glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);

//for(int r = 0; r < 4; r++)
//{
//    for(int c = 0; c < 4; c++)
//    {
//        std::cout<<projectionMatrix[r*4+c]<<" ";
//    }
//    std::cout<<std::endl;
//}


//float modelviewMatrix[16];
//glGetFloatv(GL_MODELVIEW_MATRIX, modelviewMatrix);
//for(int c = 0; c < 4; c++)
//{
//    for(int r = 0; r < 4; r++ )
//    {
//        std::cout<<modelviewMatrix[r*4+c]<<" ";
//    }
//    std::cout<<std::endl;
//}

