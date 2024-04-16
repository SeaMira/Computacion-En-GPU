#include<iostream>
#include "../include/matrix/Matrix.h"
#include <stdexcept>

int Matrix::get_n() const {
    return this-> n;
}

int Matrix::get_m() const {
    return this-> m;
}

double Matrix::get_element(int i, int j) const {
    if (i < 0 || i >= n || j < 0 || j >= m) throw std::invalid_argument("Índice de matriz no permitido");
    return mat[i*m+j];
}

Matrix::~Matrix() {}

Matrix::Matrix() {
    mat = nullptr;
}

Matrix::Matrix(int N) {
    if (N <= 0) throw std::invalid_argument("Dimensión de matriz no permitida");
    this->mat = std::make_unique<double[]>(N);
    this->n = 1; 
    this->m = N;
    for (int i = 0; i < N; i++) {
        this->mat[i] = 0;
    }
}

Matrix::Matrix(int N, int M) {
    if (N <= 0 || M <= 0) throw std::invalid_argument("Dimensiones de matriz no permitida");
    this->mat = std::make_unique<double[]>(N*M);
    this->n = N; this->m = M;
    for (int i = 0; i < N*M; i+=M) {
        for (int j = 0; j < M; j++) {
            this->mat[i + j] = 0;
        } 
    }
}

Matrix::Matrix(const std::string &filename) {
    std::string strn, strm; 
    std::ifstream newfile(filename);
    if (!newfile.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo");
    }
    getline(newfile, strn);
    getline(newfile, strm);
    if (stoi(strn) <= 0 || stoi(strm) <= 0) throw std::logic_error("Dimensiones de matriz no permitida");
    this->n = stoi(strn); this->m = stoi(strm);
    mat = std::make_unique<double[]>((this->n)*(this->m));
    std::string strvalue;
    for (int i = 0; i < n*m; i+=m) {
        for (int j = 0; j < m; j++) {
            getline(newfile, strvalue);
            mat[i + j] = stod(strvalue);
        } 
    }
    newfile.close();
}

Matrix::Matrix(const Matrix & matrix) {
    int N = matrix.get_n(), M = matrix.get_m();
    this->n = N;
    this->m = M;
    this->mat = std::make_unique<double[]>(N*M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            this->mat[i*M + j] = matrix.get_element(i, j);
        } 
    }
}

double& Matrix::operator[](std::size_t x, std::size_t y) { // Set value to (i,j) <row,column>
if (x < 0 || x >= n || y < 0 || y >= m) throw std::out_of_range("Índice de matriz no permitido");
    return mat[x*m + y];  
}

const double& Matrix::operator[](std::size_t x, std::size_t y) const {
    if (x < 0 || x >= n || y < 0 || y >= m) throw std::out_of_range("Índice de matriz no permitido");
    return mat[x*m + y];
}

void Matrix::fill(double value){
    for (int i = 0; i < n*m; i+=m) {
        for (int j = 0; j < m; j++) {
            this->mat[i + j] = value;
        } 
    }
}

std::tuple<int, int> Matrix::size() const {
    return std::tuple<int, int>(this->n, this->m);
}

int Matrix::length() const {
    int N = this->n;
    if (this->m < N) return N;
    return this->m;
}

double Matrix::max() const { // Maximum value of the matrix
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
double Matrix::min() const { // Maximum value of the matrix
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

std::ostream& operator<<(std::ostream &os, const Matrix &mat) {
    int N = mat.get_n(), M = mat.get_m();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            os << mat.get_element(i, j) << " ";
        } 
        os << "\n";
    }
    return os;
}

void Matrix::save_to_file(const std::string &filename) const {
    std::ofstream newfile(filename);
    newfile << this->n << "\n";
    newfile << this->m << "\n";
    for (int i = 0; i < n*m; i+=m) {
        for (int j = 0; j < m; j++) {
            newfile << this->mat[i+j] << "\n";
        } 
    }
}

bool Matrix::operator==(const Matrix &matrix) const {
    if (this->n != matrix.get_n() || this->m != matrix.get_m()) return false;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (this->mat[i*m+j] != matrix.get_element(i, j)) return false;
        } 
    }
    return true;
}

bool Matrix::operator!=(const Matrix &matrix) const {
    if (this->n != matrix.get_n() || this->m != matrix.get_m()) return true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (this->mat[i*m+j] != matrix.get_element(i, j)) return true;
        } 
    }
    return false;
}

Matrix& Matrix::operator=(const Matrix &matrix) {
    this->n = matrix.get_n();
    this->m = matrix.get_m();
    this->mat = std::make_unique<double[]>(n*m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            this->mat[i*m+j] = matrix.get_element(i, j);
        } 
    }
    return *this;
}

Matrix& Matrix::transpose() {
    std::unique_ptr<double[]> new_mat = std::make_unique<double[]>(n*m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            new_mat[j*n+i] = this->mat[i*m+j];            
        } 
    }
    int k = this->m;
    this->m = this->n;
    this->n = k;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            this->mat[i*m+j] = new_mat[i*m+j];         
        } 
    }
    return *this;
}

Matrix& Matrix::operator*=(const Matrix &matrix) { // Multiplication
    if (this->m != matrix.get_n()) {
        throw std::logic_error("Cantidad de columnas de matriz izquierda no coincide con cantidad de filas de matriz derecha.");
    }
    std::unique_ptr<double[]> new_mat = std::make_unique<double[]>(this->n * matrix.get_m());
    for (int i = 0; i < this->n; i++) {
        for (int j = 0; j < matrix.get_m(); j++) {
            double new_value = 0;
            for (int k = 0; k < this->m; k++) {
                new_value += this->mat[i*this->m + k] * matrix.get_element(k, j);
            }
            new_mat[i*matrix.get_m()+j] = new_value;
        }
    }
    this->m = matrix.get_m();
    this->mat.swap(new_mat);
    return *this;
}

Matrix& Matrix::operator*=(double a){ // Multiply by a constant
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            this->mat[i*m+j] *= a;
        } 
    }
    return *this;
}

Matrix& Matrix::operator+=(const Matrix &matrix) { // Add
    if (this->n != matrix.get_n() || this->m != matrix.get_m()) {
        throw std::logic_error("Dimensiones de matrices no coinciden");
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            this->mat[i*m+j] += matrix.get_element(i,j);
        } 
    }
    return *this;
}
Matrix& Matrix::operator-=(const Matrix &matrix) { // Substract
    if (this->n != matrix.get_n() || this->m != matrix.get_m()) {
        throw std::logic_error("Dimensiones de matrices no coinciden");
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            this->mat[i*m+j] -= matrix.get_element(i,j);
        } 
    }
    return *this;
}
