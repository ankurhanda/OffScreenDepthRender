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

//    int start=15;

    for(int i = 1 ; i < 12; i++ )
    {
        char fileName[100];

        sprintf(fileName,"/home/ankur/livingroom/previous_mtls/livingroom%d.mtl",i);

        ifstream ifile(fileName);

        sprintf(fileName,"livingroom%d.mtl",i);

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
//                    ofile << "map_Kd ../texture_library/duvet/5634339c2263c3d69c2b4838bd5d80cc.jpg" << std::endl;
//                    ofile << std::endl;

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

//                    for(int i = 0; i < filesinDir.size(); i++)
//                    {
//                        std::cout<<filesinDir.at(i) << std::endl;
//                    }
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
                       std::cout<<filesinDir.at(random_texture) << std::endl;
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

//    ifstream ifile("/home/ankur/office/office12.obj");

//    ofstream ofile("office12_new.obj");

//    ofstream ofile_mat("office12_new.mtl");


//    char readlinedata[300];

//    string _o,_matname;

//    if ( ifile.is_open() )
//    {
//        while(1)
//        {
//            ifile.getline(readlinedata,300);

//            if ( ifile.eof() )
//                break;

//            istringstream iss(readlinedata);

//            string string_line = iss.str();

//            if  ( string_line.find("o ") != std::string::npos )
//            {
//                iss >> _o;
//                iss >> _matname;
//            }
//            if  ( string_line.find("usemtl ") != std::string::npos)
//            {

//                ofile<<"usemtl " << _matname<< std::endl;

//                ofile_mat << "newmtl "<< _matname << std::endl;
//                ofile_mat << "Ns 96.078431" << std::endl;
//                ofile_mat << "Ka 0.000000 0.000000 0.000000" << std::endl;
//                ofile_mat << "Kd 0.640000 0.640000 0.640000" << std::endl;
//                ofile_mat << "Ks 0.500000 0.500000 0.500000" << std::endl;
//                ofile_mat << "Ni 1.000000" << std::endl;
//                ofile_mat << "d 1.000000" << std::endl;
//                ofile_mat << "illum 2" << std::endl;
//                ofile_mat << std::endl;
//            }

//            if  ( string_line.find("usemtl ") == std::string::npos)
//            {
//                ofile << string_line<<std::endl;
//            }

////            cout << string_line.substr(0,string_line.find("_Archinteriors")) << std::endl;

////            string replacement = string_line.substr(0,string_line.find("_Archinteriors"));

////            ofile <<" sed -i 's/"<<string_line<<"/"<< replacement << "/g' office12_objects.txt " << std::endl;
//        }
//    }

//    ofile.close();
//    ifile.close();
//    ofile_mat.close();

}
