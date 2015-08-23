#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>

using namespace std;

int main(void)
{
    std::vector<std::string> objects;
//    objects.push_back("desk");
//    objects.push_back("chair");
    objects.push_back("table");
    objects.push_back("night_stand");
    objects.push_back("sofa");
    objects.push_back("bed");

    int n_objects_max = 1000;

    for (int o=0; o<objects.size(); ++o)
    {
        char objdirname[400];
        sprintf (objdirname,"/media/viorik/Work/ModelNet/ModelNet10/%s/trainOBJ/",objects[o].c_str());
        char command[400];
        sprintf(command,"mkdir %s",objdirname);
        system(command);
        for(int i = 1; i <= n_objects_max; i++ )
        {
            char pathfile[400];

            sprintf(pathfile,"/media/viorik/Work/ModelNet/ModelNet10/%s/train/%s_%04d.off",objects[o].c_str(),objects[o].c_str(),i);

            //std::string pathFile(pathfile);

            ifstream ifile(pathfile);

            if ( !boost::filesystem::exists( pathfile) )
            {
                std::cerr<<"File: "<< pathfile << " does not exist" << std::endl;

                ifile.close();

                break;//exit(1);
            }

            char readlinedata[300];

            ifile.getline(readlinedata,200);

            istringstream iss(readlinedata);

            std::string str_off;

            iss >> str_off;

            int numVertices, numFaces, numEdges;

            ifile.getline(readlinedata,200);

            istringstream vss(readlinedata);

            vss >> numVertices;

            vss >> numFaces;

            vss >> numEdges;

            std::cout<<str_off<<" "<<numVertices<<" "<<numFaces<<" "<<numEdges<<std::endl;

            sprintf(pathfile,"/media/viorik/Work/ModelNet/ModelNet10/%s/trainOBJ/%s_%04d.obj",objects[o].c_str(),objects[o].c_str(),i);

            ofstream ofile(pathfile);

            std::cout<<pathfile<<std::endl;

            int count = 0;

            while(1)
            {
                if(ifile.eof() )
                    break;

                ifile.getline(readlinedata,300);

                istringstream vss(readlinedata);

                if ( count < numVertices )
                {
                    float x,y,z;

                    vss >> x; vss >> y; vss >> z;

                    ofile<<"v "<< x <<" "<<y <<" "<<z << std::endl;
                }

                if ( count >= numVertices && count-numVertices < numFaces )
                {
                    int points, v1, v2, v3;

                    vss >> points;

                    vss >> v1;

                    vss >> v2;

                    vss >> v3;

                    ofile <<"f "<<v1+1<<" "<<v2+1<<" "<<v3+1<<std::endl;
                }

                count++;
            }

            ofile.close();

            ifile.close();

        }
    }
}
