#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <assert.h>

class MNISTLoader {
private:
    unsigned char *m_images;
    int* m_labels;
    int m_size;
    int m_rows;
    int m_cols;

    inline unsigned int to_int(char* p) const {
        return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
                ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
    }

    void load_images(std::string image_file, int num) {
        std::FILE* f = std::fopen(image_file.c_str(), "rb");
        char p[4];

        std::fread(p, sizeof(p[0]), 4, f);
        int magic_number = to_int(p);
        assert(magic_number == 0x803);

        std::fread(p, sizeof(p[0]), 4, f);
        m_size = to_int(p);
        // limit
        if (num != 0 && num < m_size) m_size = num;

        std::fread(p, sizeof(p[0]), 4, f);
        m_rows = to_int(p);

        std::fread(p, sizeof(p[0]), 4, f);
        m_cols = to_int(p);
        m_images = new unsigned char[m_size * m_rows * m_cols];

        for (int i = 0; i < m_size; ++i) {
            std::fread(&m_images[i*m_rows*m_cols], sizeof(unsigned char), m_rows * m_cols, f);
        }
        std::fclose(f);
    }

    void load_labels(std::string label_file, int num) {
        std::FILE* f = std::fopen(label_file.c_str(), "rb");
        char p[4];

        std::fread(p, sizeof(p[0]), 4, f);
        int magic_number = to_int(p);
        assert(magic_number == 0x801);

        std::fread(p, sizeof(p[0]), 4, f);
        int size = to_int(p);
        
        if (num != 0 && num < size) size = num;
        m_labels = new int[size];
        for (int i=0; i<size; ++i) {
            std::fread(p, sizeof(p[0]), 1, f);
            m_labels[i] = p[0];
        }
        std::fclose(f);
    }   

public:

    MNISTLoader(std::string image_file, std::string label_file, int num) 
        : m_size(0), m_rows(0), m_cols(0) {
        load_images(image_file, num);
        load_labels(label_file, num);
    }

    MNISTLoader(std::string image_file, std::string label_file) 
        : MNISTLoader(image_file, label_file, 0) {}
    
    MNISTLoader(){}

    /*
        Deconstructor leads to segmentation fault when:
            - trying to access loaderx[0]
            - at the end of the programm
    */
    // ~MNISTLoader() {
    //     delete m_labels;
    //     delete m_images;
    // }

    int size() { return m_size; }
    int rows() { return m_rows; }
    int cols() { return m_cols; }

    unsigned char * const images(int id) { 
        return &m_images[id*m_rows*m_cols]; 
    }

    int labels(int id) { 
        return m_labels[id]; 
    }
};

#endif