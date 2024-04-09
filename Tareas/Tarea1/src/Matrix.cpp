#include<iostream>
#include "../include/matrix/Matrix.h"

class Matrix {
    private:
        std::unique_ptr<double[]> mat; // Store the matrix
        int n = 0; // Number of rows
        int m = 0; // Number of columns

    public:

        int get_n() const {
            return this-> n;
        }

        int get_m() const {
            return this-> m;
        }

        double get_element(int i, int j) const {
            return mat[i*m+j];
        }

        Matrix() {
            // mat = std::unique_ptr<double[]>(new double(n*m));
            // for (int i = 0; i < n*m; i+=m) {
            //     for (int j = 0; j < m; j++) {
            //         mat[i + j] = 0;
            //     } 
            // }
        }

        Matrix(int n) {
            mat = std::unique_ptr<double[]>(new double(n));
            this->n = 1; this->m = n;
            for (int i = 0; i < n; i++) {
                mat[i] = 0;
            }
        }

        Matrix(int n, int m) {
            mat = std::unique_ptr<double[]>(new double(n*m));
            this->n = n; this->m = m;
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    mat[i + j] = 0;
                } 
            }
        }

        Matrix(const std::string &filename) {
            std::string strn, strm; 
            std::ifstream newfile(filename);
            if (!newfile.is_open()) {
                throw std::runtime_error("No se pudo abrir el archivo");
            }
            getline(newfile, strn);
            getline(newfile, strm);
            this->n = stoi(strn); this->m = stoi(strm);
            mat = std::unique_ptr<double[]>(new double(n*m));
            std::string strvalue;
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    getline(newfile, strvalue);
                    mat[i + j] = stod(strvalue);
                } 
            }
            newfile.close();
        }

        Matrix(const Matrix & matrix) {
            this->n = matrix.get_n();
            this->m = matrix.get_m();
            this->mat = std::unique_ptr<double[]>(new double(n*m));
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    this->mat[i + j] = matrix.get_element(i, j);
                } 
            }
        }

        double& operator[](std::size_t x, std::size_t y) { // Set value to (i,j) <row,column>
            return mat[x*m + y];  
        }

        const double& operator[](std::size_t x, std::size_t y) const {
            return this->get_element(x, y);
        }

        void fill(double value){
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    this->mat[i + j] = value;
                } 
            }
        }

        std::tuple<int, int> &size() const {
            return std::tuple<int, int>(this->n, this->m);
        }

        double max() const { // Maximum value of the matrix
            double max = this->mat[0];
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    if (max < this->mat[i+j]) {
                        max = this->mat[i+j];
                    }
                } 
            }
            return max;
        }
        double min() const { // Maximum value of the matrix
            double min = this->mat[0];
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    if (min > this->mat[i+j]) {
                        min = this->mat[i+j];
                    }
                } 
            }
            return min;
        }

        friend std::ostream& operator<<(std::ostream &os, const Matrix &mat) {
            int N = mat.get_n(), M = mat.get_m();
            for (int i = 0; i < N*M; i+=M) {
                for (int j = 0; j < M; j++) {
                    os << mat.get_element(i, j) << " ";
                } 
                os << "\n";
            }
            return os;
        }

        void save_to_file(const std::string &filename) const {
            std::ofstream newfile(filename);
            newfile << this->n << "\n";
            newfile << this->m << "\n";
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    newfile << this->mat[i+j] << "\n";
                } 
            }
        }

        bool operator==(const Matrix &matrix) const {
            if (this->n != matrix.get_n() || this->m != matrix.get_m()) return false;
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    if (this->mat[i+j] != matrix.get_element(i, j)) return false;
                } 
            }
            return true;
        }

        bool operator!=(const Matrix &matrix) const {
            if (this->n != matrix.get_n() || this->m != matrix.get_m()) return true;
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    if (this->mat[i+j] != matrix.get_element(i, j)) return true;
                } 
            }
            return false;
        }

        Matrix &operator=(const Matrix &matrix) {
            this->n = matrix.get_n();
            this->m = matrix.get_m();
            this->mat = std::unique_ptr<double[]>(new double(n*m));
            for (int i = 0; i < n*m; i+=m) {
                for (int j = 0; j < m; j++) {
                    this->mat[i+j] = matrix.get_element(i, j);
                } 
            }
        }

        Matrix &transpose() {
            for (int i = 0; i < n*m; i++) {
                for (int j = 0; j < m; j++) {
                    double temp = this->mat[j*n+i];
                    this->mat[j*n+i] = this->mat[i*m+j];
                    this->mat[i*m+j] = temp;
                } 
            }
            int k = this->m;
            this->m = this->n;
            this->n = k;
        }

        Matrix &operator*=(const Matrix &matrix) { // Multiplication
            if (this->m != matrix.get_n()) {
                //throw exception
            }
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < matrix.get_m(); j++) {
                    double new_value = 0;
                    for (int k = 0; k < this->m; k++) {
                        new_value += this->mat[i*this->m + k] * matrix.get_element(k, j);
                    }
                    this->mat[i*this->m + j] = new_value;
                }
            }
        }

        Matrix &operator*=(double a){ // Multiply by a constant
            for (int i = 0; i < n*m; i++) {
                for (int j = 0; j < m; j++) {
                    this->mat[i*m+j] *= a;
                } 
            }
        }
        Matrix &operator+=(const Matrix &matrix) { // Add
            for (int i = 0; i < n*m; i++) {
                for (int j = 0; j < m; j++) {
                    this->mat[i*m+j] += matrix.get_element(i,j);
                } 
            }
        }
        Matrix &operator-=(const Matrix &matrix) { // Substract
            for (int i = 0; i < n*m; i++) {
                for (int j = 0; j < m; j++) {
                    this->mat[i*m+j] -= matrix.get_element(i,j);
                } 
            }
        }
};