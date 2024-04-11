#include "gtest/gtest.h"
#include "matrix/Matrix.h"


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
}

TEST(MATRiXLib, MatrixNewMatrixByReference) {
  Matrix m1(5, 5);
  Matrix m2(m1);
  

  ASSERT_EQ(m2.get_n(), 5);
  ASSERT_EQ(m2.get_m(), 5);
  ASSERT_EQ(m2.get_element(4, 4), 0);
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
}


TEST(MATRiXLib, MatrixScalarMultiplication) {
  Matrix m1(2, 4);
  
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
}
