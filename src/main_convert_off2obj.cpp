#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>

using namespace std;

int main(void)
{
    std::string object = "desk";

    for(int i = 1; i <= 889; i++ )
    {
        char filename[400];

        sprintf(filename,"/home/ankur/workspace/code/ModelNet/ModelNet10/%s/train/%s_%04d.off",object.c_str(),
                object.c_str(),i);

        std::string fileName(filename);

        ifstream ifile(fileName.c_str());

        if ( !boost::filesystem::exists( fileName) )
        {
            std::cerr<<"File: "<< fileName << " does not exist" << std::endl;

            ifile.close();

            exit(1);
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

        sprintf(filename,"/home/ankur/workspace/code/ModelNet/ModelNet10/%s/train/%s_%04d.obj",object.c_str(),
                object.c_str(),i);

        ofstream ofile(filename);

        std::cout<<filename<<std::endl;

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
