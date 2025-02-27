#ifndef LBM_IO_
#define LBM_IO_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sys/stat.h>
#include "LBM_binary.H"

using namespace std;

/*  
    [slice_index]:  vector which contains the index of each sliced plane;
    [c_dim]:        dimension char = 'x', 'y' or 'z'. Indicate that the sliced plane is perpendicular to axis [c_dim];
    [file_path]:  multifab folder directory, it's ending char can neglect the dash line;
    [file_mfolder]:   multifab folder name which used for storing the sliced 2D plot of [mfab];
                    it's ending char can neglect the dash line;
    [file_prefix]:  each sliced 2D plot file prefix;
    [ncomp]:        component index of [mfab] we hope to write out;
    Write one component each time and all sliced plane data are stored under the directory: file_path/file_mfolder/file_prefix*;
    For each sliced plane, the data are written out to file according to the C-Format Ordering; (row, col) = (j, k), (i, k) or (i, j);
*/
void SliceWriteMultiFabCompToPlainText(char c_dim, vector<int> slice_index, MultiFab& mfab, int ncomp, // can only use the reference & for MultiFab;
    string file_path, string file_mfolder = "MFab", string file_prefix = "mfab", char delimiter = '\t', bool ifShowDetails = true){
    // check last char of file_path, if it contains no '/', then add it;
    if(file_path[file_path.size()-1]!='/'){
        file_path = file_path + "/";
    }
    if(file_mfolder[file_mfolder.size()-1]=='/'){
        file_mfolder = file_mfolder.substr(0, file_mfolder.size()-1);
    }
    string mfolder_dir = file_path+file_mfolder;
    while(filesystem::exists(mfolder_dir)){
        mfolder_dir = mfolder_dir + ".New";
    }
    mfolder_dir = mfolder_dir + "/";

    /*for(char& c : file_path) {
        do_things_with(c);
    }*/

    // build new 3D tensor for writing;
    // first obtian the mfab geometry size;
    int Nx = 0; int Ny = 0; int Nz = 0;
    for(MFIter mfi(mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        //const amrex::Array4<Real>& mf_array = mf.array(mfi);
        ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k){
            if(i>=Nx){ Nx = i; }
            if(j>=Ny){ Ny = j; }
            if(k>=Nz){ Nz = k; }
        });
    }
    Nx++;   Ny++;   Nz++;
    // build new 3D tensor for writing;
    Real*** sliced_mfab = new Real**[Nx];    // begin memory allocation
    for(int nx = 0; nx < Nx; nx++) {
        sliced_mfab[nx] = new Real*[Ny];
        for(int ny = 0; ny < Ny; ny++) {
            sliced_mfab[nx][ny] = new Real[Nz];
            for(int nz = 0; nz < Nz; nz++) { // initialize the memories;
                sliced_mfab[nx][ny][nz] = 0;
            }
        }
    }
    // copy values from mfab to 3D tensor
    auto const & mfab_multi_array4 = mfab.arrays(); 
    ParallelFor(mfab, [&] AMREX_GPU_DEVICE(int nbx, int i, int j, int k){
        for(vector<int>::iterator idx = slice_index.begin(); idx != slice_index.end(); idx++){
            if((c_dim=='x'||c_dim=='X')&&i==(*idx)){
                sliced_mfab[i][j][k] = mfab_multi_array4[nbx](i,j,k,ncomp);       
            }
            if((c_dim=='y'||c_dim=='Y')&&j==(*idx)){
                sliced_mfab[i][j][k] = mfab_multi_array4[nbx](i,j,k,ncomp);       
            }
            if((c_dim=='z'||c_dim=='Z')&&k==(*idx)){
                sliced_mfab[i][j][k] = mfab_multi_array4[nbx](i,j,k,ncomp);       
            }
        }
    });
    
    // create mfab folder, depending on if it exists, original or folder with suffix will be created;
    if (mkdir(mfolder_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0){
        if(ifShowDetails){
                Print() << "Folder " << mfolder_dir << " has been created" << '\n'; 
        } else {
            std::cout << "cannot create MultiFab data folder error:" << strerror(errno) << std::endl;
            throw std::runtime_error( strerror(errno) );
        }
    }

    ofstream myfile;
    for (vector<int>::iterator idx = slice_index.begin(); idx != slice_index.end(); idx++){
        string filename = mfolder_dir + file_prefix + to_string(*idx);
        myfile.open(filename,fstream::out);
        if(myfile.is_open()){
            if((c_dim=='x'||c_dim=='X')){
                for(int ny=0; ny<Ny; ny++){
                    for(int nz=0; nz<Nz; nz++){
                        myfile << sliced_mfab[*idx][ny][nz] << delimiter; 
                    }
                    myfile << endl;
                }        
            }
            if((c_dim=='y'||c_dim=='Y')){
                for(int nx=0; nx<Nx; nx++){
                    for(int nz=0; nz<Nz; nz++){
                        myfile << sliced_mfab[nx][*idx][nz] << delimiter; 
                    }
                    myfile << endl;
                }        
            }
            if((c_dim=='z'||c_dim=='Z')){
                for(int nx=0; nx<Nx; nx++){
                    for(int ny=0; ny<Ny; ny++){
                        myfile << sliced_mfab[nx][ny][*idx] << delimiter; 
                    }
                    myfile << endl;
                }        
            }
            myfile.close();
        }else{
            cout << "Fail to create file " << filename << endl;
            exit(0);
        }
    }

    for(int nx=0; nx<Nx; nx++){
        for(int ny=0; ny<Ny; ny++){
            delete[] sliced_mfab[nx][ny];
        }
        delete[] sliced_mfab[nx];
    }
    delete[] sliced_mfab;

}

/*
    Compared with funcion [SliceWriteMultiFabCompToPlainText], it provides the option to write multiple components at one time;
    [vec_comps]:    vector of components' index we hope to write out; 
    Data storage structure is:  file_path/file_mfolder/comp[?]/file_prefix*; Each component is stored in its own seperated folder;
*/
void SliceWriteMultiFabToPlainText(char c_dim, vector<int> slice_index, MultiFab& mfab, vector<int> vec_comps, // can only use the reference & for MultiFab;
    string file_path, string file_mfolder = "MFab", string file_prefix = "mfab", char delimiter = '\t', bool ifShowDetails = true){
    // check last char of file_path, if it contains no '/', then add it;
    if(file_path[file_path.size()-1]!='/'){
        file_path = file_path + "/";
    }
    if(file_mfolder[file_mfolder.size()-1]=='/'){
        file_mfolder = file_mfolder.substr(0, file_mfolder.size()-1);
    }
    string mfolder_dir = file_path+file_mfolder;
    while(filesystem::exists(mfolder_dir)){
        mfolder_dir = mfolder_dir + ".New";
    }
    mfolder_dir = mfolder_dir + "/";
    // create mfab folder, depending on if it exists, original or folder with suffix will be created;
    if (mkdir(mfolder_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0){    // -1 for unsuccessful;
        if(ifShowDetails){
            Print() << "Folder " << mfolder_dir << " has been created" << '\n'; 
        } else {
            std::cout << "cannot create MultiFab data folder error:" << strerror(errno) << std::endl;
            throw std::runtime_error( strerror(errno) );
        }
    }
    
    /* Here we utilize the feature: file_path/              file_mfolder/   file_prefix*;
                                    file_path/file_mfolder/ comp[?]/        file_prefix*;
        replace "file_path/" with "file_path/file_mfolder/" and "file_mfolder/" with "comp[?]/", they are at the same position;
    */
    for(vector<int>::iterator ncomp = vec_comps.begin(); ncomp != vec_comps.end(); ncomp++){
        string comp_dir = "comp" + to_string(*ncomp);
        SliceWriteMultiFabCompToPlainText(c_dim, slice_index, mfab, (*ncomp), mfolder_dir, comp_dir,
        file_prefix, delimiter, ifShowDetails);
    }
}

// write whole MultiFab variable to plain text file
void WriteMultiFabToPlainText(char c_dim, MultiFab& mfab, // can only use the reference & for MultiFab;
    string file_path, string file_mfolder = "MFab", string file_prefix = "mfab", char delimiter = '\t', bool ifShowDetails = true){
    
    // first obtian the mfab geometry size & number of components
    int Nx = 0; int Ny = 0; int Nz = 0;
    auto const & mfab_multi_array4 = mfab.arrays();
    Array4<Real> const mfab_array4 = mfab_multi_array4[0];
    for(MFIter mfi(mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        //const amrex::Array4<Real>& mf_array = mf.array(mfi);
        ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k){
            if(i>=Nx){ Nx = i; }
            if(j>=Ny){ Ny = j; }
            if(k>=Nz){ Nz = k; }
        });
    }
    Nx++;   Ny++;   Nz++;
    int nComps = mfab_array4.nComp();
    int nSlicedPlanes = 0;
    vector<int> vec_sliced_index_set;

    switch(c_dim) {
        case 'x':
        case 'X':
            nSlicedPlanes = Nx;
            break;
        case 'y':
        case 'Y':
            nSlicedPlanes = Ny;
            break;
        case 'z':
        case 'Z':
            nSlicedPlanes = Nz;
    }
    if(nSlicedPlanes != 0){
        for(int i=0; i<nSlicedPlanes; i++){
            vec_sliced_index_set.push_back(i);
        }
    }
    vector<int> vec_comp_set;
    for(int i=0; i<nComps; i++){
        vec_comp_set.push_back(i);
    }
    SliceWriteMultiFabToPlainText(c_dim, vec_sliced_index_set, mfab, vec_comp_set,
    file_path, file_mfolder, file_prefix, delimiter, ifShowDetails);

}

/*
    [mfab]: it must be INITIALIZED before passed in; with same size of the data file;
    [c_dim]: see function "SliceWriteMultiFabCompToPlainText"; It indicates how MultiFab is stored in files;
*/
void SliceReadMultiFabCompFromPlainText(char c_dim, vector<int> slice_index, MultiFab& mfab, int ncomp, // can only use the reference & for MultiFab;
    string file_path, string file_mfolder = "MFab", string file_prefix = "mfab", char delimiter = '\t', bool ifShowDetails = true){
    // check last char of file_path, if it contains no '/', then add it;
    if(file_path[file_path.size()-1]!='/'){
        file_path = file_path + "/";
    }
    string mfolder_dir = file_path+file_mfolder;
    if(mfolder_dir[mfolder_dir.size()-1]!='/'){
        mfolder_dir = mfolder_dir + "/";
    }
    int slice_index_num = slice_index.size();

    // first get the number columns from the 1st line of the first sliced plane file;
    string sliced_mfab_dir = mfolder_dir + file_prefix + to_string(slice_index[0]);
    ifstream inputFile(sliced_mfab_dir);
    if (!inputFile) {
        cerr << "Error opening file!" << endl;
        exit(0);
    }
    string strline; bool tag = true;    int numRow = 0; int numCol = 0;
    while(getline(inputFile, strline)){
        if(tag){
            istringstream iss(strline);
            vector<Real> vec_row;   Real valtmp;
            // Extract integers from the line and store them in a row vector
            while (iss >> valtmp) {
                vec_row.push_back(valtmp);
            }
            numCol = vec_row.size();
            tag = false;
        }
        numRow++;
    }
    inputFile.close();


    Real** slice_buffer = new Real* [numRow];
    for(int nr=0; nr<numRow; nr++){
        slice_buffer[nr] = new Real[numCol];
    }
    for(int idx=0; idx<slice_index_num; idx++){
        sliced_mfab_dir = mfolder_dir + file_prefix + to_string(slice_index[idx]);
        ifstream inputFile(sliced_mfab_dir);
        if (!inputFile) {
            cerr << "Error opening file!" << endl;
            exit(0);
        }
        int nrow = 0;   int ncol = 0;
        while(getline(inputFile, strline)){
            istringstream iss(strline); Real valtmp;
            // Extract integers from the line and store them in a row vector
            while (iss >> valtmp) {
                slice_buffer[nrow][ncol] = valtmp;
                ncol++;
            }
            nrow++; ncol = 0;
        }
        inputFile.close();

        //  assign the sliced matrix to [mfab]; (row, col) = (j, k), (i, k) or (i, j);
        auto const& mfab_multi_array4 = mfab.arrays();
        ParallelFor(mfab, [&] AMREX_GPU_DEVICE(int nbx, int i, int j, int k){
            if((c_dim=='x'||c_dim=='X')&&i==slice_index[idx]){
                mfab_multi_array4[nbx](i,j,k,ncomp) = slice_buffer[j][k];       
            }
            if((c_dim=='y'||c_dim=='Y')&&j==slice_index[idx]){
                mfab_multi_array4[nbx](i,j,k,ncomp) = slice_buffer[i][k];       
            }
            if((c_dim=='z'||c_dim=='Z')&&k==slice_index[idx]){
                mfab_multi_array4[nbx](i,j,k,ncomp) = slice_buffer[i][j];       
            }
        });
    }
    // free buffer memories;
    for(int nr=0; nr<numRow; nr++){
        delete[] slice_buffer[nr];
    }
    delete[] slice_buffer;
}

//  read in multiple components from files into [mfab];
//  [mfab]: similarly, it must be INITIALIZED before passed in; with same size of the data file;
void SliceReadMultiFabFromPlainText(char c_dim, vector<int> slice_index, MultiFab& mfab, vector<int> vec_comps, // can only use the reference & for MultiFab;
    string file_path, string file_mfolder = "MFab", string file_prefix = "mfab", char delimiter = '\t', bool ifShowDetails = true){
    
    // check last char of file_path, if it contains no '/', then add it;
    if(file_path[file_path.size()-1]!='/'){
        file_path = file_path + "/";
    }
    if(file_mfolder[file_mfolder.size()-1]!='/'){
        file_mfolder = file_mfolder + "/";
    }
    string mfolder_dir = file_path+file_mfolder;    // [mfab] data folder directory;

    /* Here we utilize the feature: file_path/              file_mfolder/   file_prefix*;
                                    file_path/file_mfolder/ comp[?]/        file_prefix*;
        replace "file_path/" with "file_path/file_mfolder/" and "file_mfolder/" with "comp[?]/", they are at the same position;
    */
    for(vector<int>::iterator ncomp = vec_comps.begin(); ncomp != vec_comps.end(); ncomp++){
        string comp_dir = "comp" + to_string(*ncomp);
        SliceReadMultiFabCompFromPlainText(c_dim, slice_index, mfab, (*ncomp), mfolder_dir, comp_dir,
        file_prefix, delimiter, ifShowDetails);
    }
}


//  Get the geometry size & number of components from Plain text MultiFab data files;
//  Each component's sub-folder name is "comp" by default;
void ReadMultiFabSizeFromPlainText(char c_dim, vector<int>& vec_params, string file_path, string file_mfolder = "MFab",
    string file_prefix = "mfab", char delimiter = '\t'){
    // check last char of file_path, if it contains no '/', then add it;
    if(file_path[file_path.size()-1]!='/'){
        file_path = file_path + "/";
    }
    if(file_mfolder[file_mfolder.size()-1]!='/'){
        file_mfolder = file_mfolder + "/";
    }
    string mfolder_dir = file_path + file_mfolder;
    int numComps = 0;   int numRow = 0; int numCol = 0; int numSlices = 0;
    string sliced_mfab_comp_dir = mfolder_dir + "comp" + to_string(numComps);
    while(filesystem::exists(sliced_mfab_comp_dir)){
        if(numComps == 0){
            string sliced_mfab_comp_planes_dir = sliced_mfab_comp_dir + "/" + file_prefix + to_string(numSlices);
            while(filesystem::exists(sliced_mfab_comp_planes_dir)){
                if(numSlices == 0){
                    ifstream inputFile(sliced_mfab_comp_planes_dir);    // read the first file under component 0;
                    if (!inputFile) {
                        cerr << "Error opening file!" << endl;
                        exit(0);
                    }
                    string strline; bool tag = true;    
                    while(getline(inputFile, strline)){
                        if(tag){
                            istringstream iss(strline);
                            vector<Real> vec_row;   Real valtmp;
                            // Extract integers from the line and store them in a row vector
                            while (iss >> valtmp) {
                                vec_row.push_back(valtmp);
                            }
                            numCol = vec_row.size();
                            tag = false;
                        }
                        numRow++;
                    }
                    inputFile.close();
                }
                numSlices++;
                sliced_mfab_comp_planes_dir = sliced_mfab_comp_dir + "/" + file_prefix + to_string(numSlices);
            }
        }
        numComps++;
        sliced_mfab_comp_dir = mfolder_dir + "comp" + to_string(numComps);
    }
    // add into the params vector;
    switch(c_dim) {
        case 'x':
        case 'X':
            vec_params.push_back(numSlices);
            vec_params.push_back(numRow);
            vec_params.push_back(numCol);
            break;
        case 'y':
        case 'Y':
            vec_params.push_back(numRow);
            vec_params.push_back(numSlices);
            vec_params.push_back(numCol);
            break;
        case 'z':
        case 'Z':
            vec_params.push_back(numRow);
            vec_params.push_back(numCol);
            vec_params.push_back(numSlices);
    }
    vec_params.push_back(numComps);
}

//  [mfab]: it must be INITIALIZED before passed in; with same size of the data file;
void ReadMultiFabFromPlainText(char c_dim, MultiFab& mfab,// can only use the reference & for MultiFab;
    string file_path, string file_mfolder = "MFab", string file_prefix = "mfab", char delimiter = '\t', bool ifShowDetails = true){
    
    vector<int> vec_params; // mfab dimension-size; 
    ReadMultiFabSizeFromPlainText(c_dim, vec_params, file_path, file_mfolder, file_prefix, delimiter);

    vector<int> slice_index;
    switch(c_dim) {
        case 'x':
        case 'X':
            for(int nx=0; nx<vec_params[0]; nx++){
                slice_index.push_back(nx);
            }
            break;
        case 'y':
        case 'Y':
            for(int ny=0; ny<vec_params[1]; ny++){
                slice_index.push_back(ny);
            }
            break;
        case 'z':
        case 'Z':
            for(int nz=0; nz<vec_params[2]; nz++){
                slice_index.push_back(nz);
            }
            break;
    }
    vector<int> vec_comps;
    for(int np=0; np<vec_params[3]; np++){
        vec_comps.push_back(np);
    }
    SliceReadMultiFabFromPlainText(c_dim, slice_index, mfab, vec_comps, file_path,
    file_mfolder, file_prefix, delimiter, ifShowDetails);
    
}

/*
    [filename]: directory of the folder which contains: folder $levelPrefix_$nlevel (e.g., Level_0) and Header file;
    Load single MultiFab variable from disk to [mfab_in];
*/
bool LoadSingleMultiFab(std::string filename, MultiFab& mfab_in, int nlevel=0,
    const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){

    if(!mfab_in.ok()) {
        std::cout << "MultiFab passed in is not defined\n";
        std::cout << "the BoxArray on the disk will be used and a new DistributionMapping will be made (called from function 'LoadSingleMultiFab').\n";
    }
    if(!filesystem::exists(filename)){
        Print() << "Cannot load in MultiFab " << filename << '\n';
        return false;
    }else{
        VisMF::Read(mfab_in, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
        Print() << "Load in MultiFab " << filename << " successfully\n";
        return true;
    }
}

/*
    [filename]: directory of the folder which contains: folder $levelPrefix_$nlevel (e.g., Level_0) and Header file;
    Load specific component [loadcomp] of MultiFab variable from disk to [mfab_in];
*/
bool LoadSlicedMultiFab(std::string filename, MultiFab& mfab_in, int loadcomp=0, int nlevel=0,
    const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){
    
    MultiFab mfab_readin;
    if(!mfab_in.ok()) {
        std::cout << "MultiFab passed in is not defined\n";
        std::cout << "the BoxArray on the disk will be used and a new DistributionMapping will be made (called from function 'LoadSlicedMultiFab').\n";
    }else{
        VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
        int ncomp = mfab_readin.nComp();
        mfab_readin.define(mfab_in.boxArray(), mfab_in.DistributionMap(), ncomp, mfab_in.nGrow());
    }
    if(!filesystem::exists(filename)){
        Print() << "Cannot load in MultiFab " << filename << '\n';
        return false;
    }else{
        VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
        Print() << "Load in MultiFab " << filename << " successfully\n";
        mfab_in.ParallelCopy(mfab_readin, loadcomp, 0, 1);    // load from srccomp [loadcomp] to destcomp [0] of [mfab_in], total 1 component;
        return true;
    }
}

/*
    [plot_file_root]:   stands for root name of plotfile; it consists of the directory name which contains all multifabs 
    and each multifab folder's prefix, i.e., dir/plt***, then plot_file_root=dir/plt; *** denotes time step info;
    [ndigits]:  number of digits 
*/
bool LoadSetOfMultiFabs(std::string plot_file_root, MultiFab& mfab_in, int step1, int step2, int plot_int,
    int loadcomp=0, bool if_print=false,
    int ndigits=5, int nlevel=0, const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){

    MultiFab mfab_readin;
    std::string filename;
    int nstep = (step2-step1)/plot_int + 1;
    if(!mfab_in.ok()) {
        std::cout << "MultiFab passed in is not defined\n";
        std::cout << "the BoxArray on the disk will be used and a new DistributionMapping will be made (called from function 'LoadSetOfMultiFabs').\n";
        filename = amrex::Concatenate(plot_file_root,step1,ndigits);
        if(!filesystem::exists(filename)){
            Print() << "Cannot load in MultiFab " << filename << '\n';
            throw runtime_error("called by function 'LoadSetOfMultiFabs'!\n");
            return false;
        }else{
            VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
            mfab_in.define(mfab_readin.boxArray(), mfab_readin.DistributionMap(), nstep, mfab_readin.nGrow());
        }
    }else{
        if(mfab_in.nComp()<nstep){
            printf("Input multifab has less components than required!\n");
            throw runtime_error("called by function 'LoadSetOfMultiFabs'!\n");
            return false;
        }
    }
    int nfile=0;
    for(int step=step1; step<=step2; step=step+plot_int){
        filename = amrex::Concatenate(plot_file_root,step,ndigits);
        if(!filesystem::exists(filename)){
            Print() << "Cannot load in MultiFab " << filename << '\n';
            throw runtime_error("called by function 'LoadSetOfMultiFabs'!\n");
            return false;
        }else{
            VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
            if(if_print){
                printf("Read in file %s ...\n", filename.c_str());
            }
            mfab_in.ParallelCopy(mfab_readin, loadcomp, nfile, 1);
            nfile++;
        }
    }
    return true;
}


void PrintMultiFab(const MultiFab& mfab, int ngrow=-1){
    int ncomp = mfab.nComp();
    auto const & mfab_ptr = mfab.arrays();

    IntVect halo;
    if(ngrow<0){
        halo = IntVect(mfab.nGrow());
    }else{
        halo = IntVect(ngrow);
    }
    ParallelFor(mfab, halo, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
        for(int c=0; c<ncomp; c++){
            printf("(%d,%d,%d,%d)-(%f)\t", i, j, k, c, mfab_ptr[nbx](i,j,k,c));
        }
        //Print() << '\n';
    });
    Print() << '\n';
}

// Variadic template to handle multiple parameters
template <typename... Args>
std::string format(const std::string& format_str, Args... args) {
    char buffer[256]; // Adjust buffer size as needed
    sprintf(buffer, format_str.c_str(), args...);
    return std::string(buffer);
}

/*
  MultiFab testfunc(ba, dm, 5, nghost);
  MultiFab testfunc_cpy(ba, dm, 5, nghost);
  
  auto const & testFunc = testfunc.arrays();
  ParallelFor(testfunc, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    for(int ncomp=0; ncomp<5; ncomp++){
      testFunc[nbx](x,y,z,ncomp) = x+y+z+ncomp;
    }
  });
  std::vector<int> g1, g2;
  for (int i = 0; i < 5; i++)
    g1.push_back(i);

  WriteMultiFabToPlainText('X', testfunc, ".", "MFab/");
  //SliceWriteMultiFabToPlainText('Y', g1, testfunc, g1, ".", "MFab/");
  //SliceReadMultiFabToPlainText('Y', g1, testfunc_cpy, g1, ".", "MFab/");
  ReadMultiFabFromPlainText('X', testfunc_cpy, ".", "MFab/");

  auto const & testFunc_cpy = testfunc_cpy.arrays();
  ParallelFor(testfunc, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,0) - testFunc[nbx](x,y,z,0) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,1) - testFunc[nbx](x,y,z,1) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,2) - testFunc[nbx](x,y,z,2) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,3) - testFunc[nbx](x,y,z,3) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,4) - testFunc[nbx](x,y,z,4) << '\t';
  });

*/


#endif