#include <iostream>
#include <iosfwd>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include "utils/map_object_label2training_label.h"

using namespace std;

void getFilesInDirectory(std::string& dirName, std::vector<std::string>& fileNames)
{
    if ( !boost::filesystem::exists(dirName))
        return;

    boost::filesystem::path targetDir(dirName);

    boost::filesystem::directory_iterator iter(targetDir), eod;

    BOOST_FOREACH(boost::filesystem::path const& i, make_pair(iter, eod))
    {
        if (is_regular_file(i))
        {
            fileNames.push_back(i.string());
        }
    }
}

int main(void)
{
    srand(time(NULL));

    for(int i = 2 ; i < 3; i++ )
    {
        char fileName[100];

        sprintf(fileName,"/home/ankur/bedrooms/previous_mtls/bedroom%d.mtl",i);

        ifstream ifile(fileName);

        sprintf(fileName,"bedroom%d.mtl",i);

        ofstream ofile(fileName);

        std::string cmd, matname;

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

                std::vector<std::string> filesinDir;
                std::string base_dir;

                if ( current_line.find("newmtl") != std::string::npos)
                {
                    iss >> cmd;
                    iss >> matname;

                    std::cout<<matname<<" -> ";
                    matname = get_class_name(matname);

                    std::transform(matname.begin(),
                                   matname.end(),
                                   matname.begin(),
                                   ::tolower);

                    std::cout<<matname<<std::endl;

                    base_dir = "/home/ankur/texture_library/" + matname;

                    getFilesInDirectory(base_dir,filesinDir);

                    for(int i = 0; i < filesinDir.size(); i++)
                    {
                        std::cout<<filesinDir.at(i) << std::endl;
                    }
                }

//                if ( current_line.empty() && previous_line.find('#') == std::string::npos)
//                {


//                }

                ofile << current_line << std::endl;

                if ( current_line.find("newmtl") != std::string::npos )
                {
                    if ( filesinDir.size())
                    {
                        int random_texture = ((float)rand()/RAND_MAX)*(filesinDir.size()-1);
                        ofile << "map_Kd "<< filesinDir.at(random_texture) << std::endl;
//                        ofile << std::endl;
                    }
                    else
                    {
                        ofile << "map_Kd ../texture_library/duvet/5634339c2263c3d69c2b4838bd5d80cc.jpg" << std::endl;
//                        ofile << std::endl;
                    }

                }

                previous_line = current_line;

            }

        }

        ifile.close();
        ofile.close();

//        std::cout<< system("cp bedroom*.mtl ~/bedrooms/") << std::endl;

    }
}
