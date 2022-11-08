#pragma once
#include <iostream>
#include <fstream>
using namespace std;

class Matrix {
private:
	int size;
	float** values;
public:
	Matrix();
	Matrix(int sz);
	Matrix(int sz, float* plain);
	~Matrix();
	Matrix(const Matrix& m);
	int getSize() const;
	float get(int i, int j) const;
	void readFromFile(ifstream& fptr);
	void setValue(int i, int j, float val);
	Matrix convolution(const Matrix& kernel);
	void saveToFile(string filename, int dtime, int fsize);
	void print();
	float* to_plain_array();

};