#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <random>

using namespace std;

void genToFile(int size, ofstream& fin, int d_range = 0, int up_range = 999) {
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> dist(d_range, up_range);

	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			fin << dist(gen) << " ";
}

int main()
{
	string fname;
	int fsize, msizeA, msizeB;
	float cf = -1;
	cout << "Enter file name > " << endl;
	cin >> fname;
	cout << "Enter size of file (Mb) > " << endl;
	cin >> fsize;
	while ((cf < 0) || (cf > 1)) {
		cout << "Enter convolution factor (0 to 1) > " << endl;
		cin >> cf;
	}

	ofstream fin(fname);
	if (!fin.is_open())
		return 1;

	/* Calculate matrix A size */
	/* Size of file in Mb divided by 4 as it takes 3 characters for number and 1 for space */
	msizeA = ceil(sqrt((fsize * 1024 * 1024) / 4));
	fin << msizeA << " ";

	/* Create matrix A with random three-digit numbers */
	genToFile(msizeA, fin);

	/* Calculate matrix B size according to convolution factor cf */
	msizeB = ceil(msizeA * cf);
	fin << msizeB << " ";

	/* Create matrix B with random three-digit numbers */
	genToFile(msizeB, fin, 0, 9);

	fin.close();

	return 0;
}