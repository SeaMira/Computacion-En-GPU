#include "gtest/gtest.h"
#include "matrix/Matrix.h"


TEST(MATRiXLib, MatrixCreate) {
  Matrix m1(5);
  Matrix m2(5, 5);
  Matrix m3();


  ASSERT_EQ(m1.get_n(), 1);
  ASSERT_EQ(m1.get_m(), 5);
  ASSERT_EQ(m1.get_element(0, 4), 0);
  ASSERT_EQ(m2.get_n(), 5);
  ASSERT_EQ(m2.get_m(), 5);
  ASSERT_EQ(m2.get_element(4, 4), 0);
  ASSERT_EQ(m3.get_n(), 0);
  ASSERT_EQ(m3.get_m(), 0);
  
}