#include "Matrix.h"

Matrix::Matrix() : size(0), values(nullptr) {}

Matrix::Matrix(int sz)
{
	size = sz;
	if (size <= 0){
		std::cout << "Error: Creating matrix with size <= 0" << size << std::endl;
		throw exception();
	}
		
	values = new float* [size];
	for (int i = 0; i < size; i++)
		values[i] = new float[size];
}

Matrix::Matrix(int sz, float* plain)
{
	size = sz;
	if (size <= 0){
		std::cout << "Error: Creating matrix with size <= 0" << size << std::endl;
		throw exception();
	}
		
	values = new float* [size];
	for (int i = 0; i < size; i++)
		values[i] = new float[size];

	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			values[i][j] = plain[i * size + j];
}

Matrix::~Matrix()
{
	for (int i = 0; i < size; i++)
		delete[] values[i];
	delete[] values;
}

Matrix::Matrix(const Matrix& m)
{
	size = m.getSize();
	values = new float* [size];
	for (int i = 0; i < size; i++)
		values[i] = new float[size];
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			values[i][j] = m.get(i, j);
}

int Matrix::getSize() const
{
	return size;
}

float Matrix::get(int i, int j) const
{
	if ((i > size) || (j > size)){
		std::cout << "Error: index out of boundaries" << std::endl;
		throw exception();
	}
	return values[i][j];
}

void Matrix::readFromFile(ifstream& fptr)
{
	if (!fptr.is_open()){
		std::cout << "Error: File not opened" << std::endl;
		throw exception();
	}
		
	fptr >> size;
	if (size <= 0){
		std::cout << "Error: size of file <=0" << std::endl;
		throw exception();
	}
	values = new float* [size];
	for (int i = 0; i < size; i++)
		values[i] = new float[size];

	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			if (!fptr.eof())
				fptr >> values[i][j];
			else {
				std::cout << "Error: file is not fully read, but matrix full" << std::endl;
				throw exception();
			}
}

void Matrix::setValue(int i, int j, float val)
{
	if ((i > size) || (j > size)){
		std::cout << "Error:(set) index out of boundaries" << std::endl;
		throw exception();
	}
	values[i][j] = val;
}

Matrix Matrix::convolution(const Matrix& kernel)
{
	int convSize = size - kernel.getSize() + 1;
	if (convSize <= 0){
		std::cout << "Error: convSize <= 0" << std::endl;
		throw exception();
	}

	Matrix convMatr(convSize);
	float sum;

	/* Moving kernel window (Matr B) through Matrix A*/
	for (int i = 0; i < convSize; ++i) {
		for (int j = 0; j < convSize; ++j) {
			/* Calculate element of convolution */
			sum = 0;
			for (int k = 0; k < kernel.getSize(); ++k) {
				for (int l = 0; l < kernel.getSize(); ++l) {
					sum += values[i + k][j + l] * kernel.get(k, l);
				}
			}
			convMatr.setValue(i, j, sum);
		}
	}

	return convMatr;
}

void Matrix::saveToFile(string filename, int dtime, int fsize)
{
	ofstream fout;
	fout.open(filename);
	if (!fout.is_open()){
		std::cout << "Error: no file to save matrix - " << filename << std::endl;
		throw exception();
	}

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j)
			fout << values[i][j] << " ";
		fout << endl;
	}
	fout << "Time: " << dtime << endl;
	fout << "Size of processed data: " << fsize;
	fout.close();
}


void Matrix::print() {
	std::cout << "Printing matrix with size: " << size << endl;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j)
			std::cout << values[i][j] << " ";
		std::cout << endl;
	}
}

float* Matrix::to_plain_array() {
	float* plain_values = new float[size * size];
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			plain_values[i * size + j] = values[i][j];

	return plain_values;
}