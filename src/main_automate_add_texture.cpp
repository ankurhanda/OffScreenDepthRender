#include<iostream>
#include<iosfwd>
#include <fstream>
#include <stdlib.h>
#include <sstream>

using namespace std;

int main(void)
{
    for(int i = 2 ; i <= 10; i++ )
    {
        char fileName[100];

        sprintf(fileName,"/home/ankur/bedrooms/bedroom%d.mtl",i);

        ifstream ifile(fileName);

        sprintf(fileName,"bedroom%d.mtl",i);

        ofstream ofile(fileName);

        if (ifile.is_open())
        {
            char readlinedata[300];

            std::string previous_line;

            while(1)
            {
                ifile.getline(readlinedata,300);

                if ( ifile.eof() )
                {
                    ofile << "map_Kd ../texture_library/duvet/5634339c2263c3d69c2b4838bd5d80cc.jpg" << std::endl;
                    ofile << std::endl;

                    break;
                }

                istringstream iss(readlinedata);

                std::string current_line(readlinedata);

                if ( current_line.find("newmtl") != std::string::npos)
                {
//                    std::cout<<current_line<<std::endl;
                    std::string cmd, matname;
                    iss >> cmd;
                    iss >> matname;

                    std::cout<<matname<<std::endl;
                }



                if ( current_line.empty() && previous_line.find('#') == std::string::npos)
                {
                    ofile << "map_Kd ../texture_library/duvet/5634339c2263c3d69c2b4838bd5d80cc.jpg" << std::endl;
                    ofile << std::endl;
                }

                ofile << current_line << std::endl;

                previous_line = current_line;

            }

        }

        ifile.close();
        ofile.close();

        std::cout<< system("cp bedroom*.mtl ~/bedrooms/") << std::endl;

    }
}
