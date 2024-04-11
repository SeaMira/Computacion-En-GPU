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