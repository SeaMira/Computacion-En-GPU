#include "gtest/gtest.h"
#include "matrix/Matrix.h"
#include <fstream>
#include <sstream>
#include <string>

TEST(MATRiXLib, MatrixGetN) {
  Matrix m1(5);
  Matrix m2(5, 5);
  Matrix m3;

  ASSERT_EQ(m1.get_n(), 1);
  ASSERT_EQ(m2.get_n(), 5);
  ASSERT_EQ(m3.get_n(), 0);
}

TEST(MATRiXLib, MatrixGetM) {
  Matrix m1(5);
  Matrix m2(5, 5);
  Matrix m3;

  ASSERT_EQ(m1.get_m(), 5);
  ASSERT_EQ(m2.get_m(), 5);
  ASSERT_EQ(m3.get_m(), 0);
}

TEST(MATRiXLib, MatrixGetElement) {
  Matrix m1(5);
  Matrix m2(5, 5);

  ASSERT_EQ(m1.get_element(0, 4), 0);
  ASSERT_EQ(m2.get_element(4, 4), 0);
  m2.fill(1);
  ASSERT_EQ(m2.get_element(4, 4), 1);
}

TEST(MATRiXLib, MatrixNewMatrixByReference) {
  Matrix m1(5, 5);
  Matrix m2(m1);
  

  ASSERT_EQ(m2.get_n(), 5);
  ASSERT_EQ(m2.get_m(), 5);
  ASSERT_EQ(m2.get_element(4, 4), 0);
}

TEST(MATRiXLib, MatrixFromFile) {
    // Crear una matriz y llenarla con valores
    Matrix m1(2, 3);
    m1[0, 0] = 1;
    m1[0, 1] = 2;
    m1[0, 2] = 3;
    m1[1, 0] = 4;
    m1[1, 1] = 5;
    m1[1, 2] = 6;
    Matrix m2("../../test_matrix_output.txt");
    ASSERT_EQ(m1, m2);

}

TEST(MATRiXLib, MatrixSetElementWithOperator) {
  Matrix m1(5);
  Matrix m2(5, 5);

  m1[0, 1] = 3.0;
  m2[1, 2] = 3.0;
  ASSERT_EQ(m1.get_element(0, 1), 3.0);
  ASSERT_EQ(m2.get_element(1, 2), 3.0);
}


TEST(MATRiXLib, MatrixGetElementWithOperator) {
  Matrix m1(5);
  Matrix m2(5, 5);

  m1[0, 1] = 3.0;
  m2[1, 2] = 3.0;
  double d1 = m1[0,1], d2 = m2[1,2];
  ASSERT_EQ(d1, 3.0);
  ASSERT_EQ(d2, 3.0);
}

TEST(MATRiXLib, MatrixFill) {
  Matrix m1(5);
  Matrix m2(5, 5);

  m1.fill(3.0);
  m2.fill(3.0);
  for (int i = 0; i < 5; i++) {
    ASSERT_EQ(m1.get_element(0, i), 3.0);
    for (int j = 0; j < 5; j++) {
      ASSERT_EQ(m2.get_element(i, j), 3.0);
    }
  }
}

TEST(MATRiXLib, MatrixGetSize) {
  Matrix m1(5);
  Matrix m2(5, 5);

  std::tuple<int, int> t1 = m1.size(), t2 = m2.size();
  ASSERT_EQ(std::get<0>(t1), 1);
  ASSERT_EQ(std::get<1>(t1), 5);
  ASSERT_EQ(std::get<0>(t2), 5);
  ASSERT_EQ(std::get<1>(t2), 5);
}

TEST(MATRiXLib, MatrixLength) {
  Matrix m1(5);
  Matrix m2(5, 5);

  ASSERT_EQ(m1.length(), 5);
  ASSERT_EQ(m2.length(), 5);
}


TEST(MATRiXLib, MatrixMax) {
  Matrix m1(5);
  Matrix m2(5, 5);

  m1[0, 1] = 3.0;
  m2[1, 2] = 3.0;
  ASSERT_EQ(m1.max(), 3.0);
  ASSERT_EQ(m2.max(), 3.0);
}

TEST(MATRiXLib, MatrixMin) {
  Matrix m1(5);
  Matrix m2(5, 5);

  m1[0, 1] = -3.0;
  m2[1, 2] = -3.0;
  ASSERT_EQ(m1.min(), -3.0);
  ASSERT_EQ(m2.min(), -3.0);
}


TEST(MATRiXLib, MatrixOutOperator) {
  Matrix m1(5);
  Matrix m2(5, 5);

  for (int i = 0; i < 5; i++) {
    m1[0, i] = i;
    for (int j = 0; j < 5; j++) {
      m2[i, j] = i*5 + j;
    }
  }

  // std::cout << m1 << "\n";
  // std::cout << m2 << "\n";  
  std::stringstream ss, ss2;
  ss << m1;
  EXPECT_EQ(ss.str(), "0 1 2 3 4 \n");
  ss2 << m2;
  EXPECT_EQ(ss2.str(), "0 1 2 3 4 \n5 6 7 8 9 \n10 11 12 13 14 \n15 16 17 18 19 \n20 21 22 23 24 \n");
}

TEST(MATRiXLib, MatrixEquivalenceOperator) {
  Matrix m1(5, 5);
  Matrix m2(5, 5);
  Matrix m3(5, 5);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      m3[i, j] = 25 - (i*5 + j);
      m1[i, j] = i*5 + j;
      m2[i, j] = i*5 + j;
    }
  }
  ASSERT_TRUE(m1 == m2);
  ASSERT_TRUE(m3 != m2);
}

TEST(MATRiXLib, MatrixAsignOperator) {
  Matrix m1(5, 5);
  Matrix m2(5, 5);
  Matrix m3(5, 5);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      m3[i, j] = 25 - (i*5 + j);
      m1[i, j] = i*5 + j;
      m2[i, j] = i*5 + j;
    }
  }
  ASSERT_TRUE(m3 != m2);
  m3 = m2;
  ASSERT_TRUE(m1 == m2);
}

TEST(MATRiXLib, MatrixTranspose) {
  Matrix m1(3, 5);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      m1[i, j] = i*5 + j;      
    }
  }
  std::cout << m1 << std::endl;
  m1.transpose();
  std::cout << m1 << std::endl;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
        double t = m1[i, j]; 
        ASSERT_EQ(t, i+j*5);
    }
  }
}

TEST(MATRiXLib, MatrixMultiplication) {
  Matrix m1(2, 4);
  Matrix m2(4, 4);
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      m1[i, j] = i*4 + j;      
    }
  }
  Matrix m3(m1);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        m2[i, j] = 1;      
      }
    }
  }
  m1 *= m2;
  ASSERT_TRUE(m1 == m3);

  Matrix m4(4, 5);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 5; j++) {
      m4[i, j] = i*4 + j;      
    }
  }
  m3 *= m4;
  Matrix m5(2, 5);
  m5[0,0] = 70;
  m5[0,1] = 76;
  m5[0,2] = 82;
  m5[0,3] = 88;
  m5[0,4] = 94;
  m5[1,0] = 190;
  m5[1,1] = 212;
  m5[1,2] = 234;
  m5[1,3] = 256;
  m5[1,4] = 278;

}


TEST(MATRiXLib, MatrixScalarMultiplication) {
  Matrix m1(2, 4);
  Matrix m2(3, 5);
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      m1[i, j] = 1;      
    }
  }
  m1 *= 2;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      ASSERT_EQ(m1.get_element(i, j),2);
    }
  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      m2[i, j] = 1;      
    }
  }
  m2 *=3;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      ASSERT_EQ(m2.get_element(i, j),3);      
    }
  }
}

TEST(MATRiXLib, MatrixAddition) {
  Matrix m1(2, 4);
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      m1[i, j] = 1;      
    }
  }
  m1 += m1;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      ASSERT_EQ(m1.get_element(i, j), 2);
    }
  }

   Matrix m2(3, 5), m3(3,5);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      m2[i, j] = i+j;   
      m3[i, j] = i+j;   
    }
  }
  m2 += m3;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      ASSERT_EQ(m2.get_element(i, j), 2*(i+j));
    }
  }
}


TEST(MATRiXLib, MatrixSubtraction) {
  Matrix m1(2, 4);
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      m1[i, j] = 1;      
    }
  }
  m1 -= m1;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      ASSERT_EQ(m1.get_element(i, j), 0);
    }
  }

  Matrix m2(3, 5), m3(3,5);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      m2[i, j] = i+j;   
      m3[i, j] = i+j;   
    }
  }
  m2 -= m3;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      ASSERT_EQ(m2.get_element(i, j), 0);
    }
  }
}


std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

TEST(MATRiXLib, MatrixSaveToFile) {
    // Crear una matriz y llenarla con valores
    Matrix m1(2, 3);
    m1[0, 0] = 1;
    m1[0, 1] = 2;
    m1[0, 2] = 3;
    m1[1, 0] = 4;
    m1[1, 1] = 5;
    m1[1, 2] = 6;

    std::string filename = "test_matrix_output.txt";

    m1.save_to_file(filename);

    std::string fileContent = readFile(filename);

    std::string expectedContent = "2\n3\n1\n2\n3\n4\n5\n6\n";
    ASSERT_EQ(fileContent, expectedContent);

    // Opcional: eliminar el archivo después de realizar el test
    // std::remove(filename.c_str());
}

// Prueba para constructor con una dimensión inválida
TEST(MATRiXLib, ConstructorInvalidDimension) {
    EXPECT_THROW({
        Matrix m(-1);
    }, std::invalid_argument);
}

// Prueba para constructor con dos dimensiones inválidas
TEST(MATRiXLib, ConstructorTwoInvalidDimensions) {
    EXPECT_THROW({
        Matrix m(-1, -2);
    }, std::invalid_argument);
}

// Prueba para constructor de archivo con dimensiones inválidas
TEST(MATRiXLib, ConstructorFromFileInvalidDimensions) {
    std::ofstream("temp.txt") << "-1\n-1\n";
    EXPECT_THROW({
        Matrix m("temp.txt");
    }, std::logic_error);
    std::remove("temp.txt");
}

// Prueba para acceso a elemento con índices inválidos
TEST(MATRiXLib, AccessElementInvalidIndex) {
    Matrix m(5, 5);
    EXPECT_THROW({
        m.get_element(-1, 0);
    }, std::invalid_argument);
    EXPECT_THROW({
        m.get_element(0, 6);
    }, std::invalid_argument);
}

// Prueba para operador [] con índice inválido
TEST(MATRiXLib, OperatorBracketInvalidIndex) {
    Matrix m(5, 5);
    EXPECT_THROW({
       ( m[0, 5] = 10);
    }, std::out_of_range);
}

// Prueba para operador += con dimensiones no coincidentes
TEST(MATRiXLib, OperatorPlusEqualsNonMatchingDimensions) {
    Matrix m1(5, 5);
    Matrix m2(5, 4);
    EXPECT_THROW({
        m1 += m2;
    }, std::logic_error);
}

// Prueba para operador -= con dimensiones no coincidentes
TEST(MATRiXLib, OperatorMinusEqualsNonMatchingDimensions) {
    Matrix m1(5, 5);
    Matrix m2(5, 4);
    EXPECT_THROW({
        m1 -= m2;
    }, std::logic_error);
}

// Prueba para multiplicación de matrices con dimensiones no coincidentes
TEST(MATRiXLib, MultiplicationNonMatchingDimensions) {
    Matrix m1(2, 3);
    Matrix m2(4, 4);
    EXPECT_THROW({
        m1 *= m2;
    }, std::logic_error);
}