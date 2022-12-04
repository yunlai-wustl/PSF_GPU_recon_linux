#ifndef IMAGE_ARRAY
#define IMAGE_ARRAY

#include "global.h"
#include "PET_geometry.h"
#include <memory.h>
#include <assert.h>


template <typename T>
class ImageArray
{
public:
	ImageArray();
	ImageArray(PET_geometry g);
	ImageArray(int dim1, int dim2, int dim3);
	~ImageArray();


	void Setup(PET_geometry g);
	void Setup(int dim1, int dim2, int dim3);
	void Reset();

	void ReadFromMem(T*& image);
	void AddFromMem(T*& image);
	void ReadFromFile(string filename);
	void WriteToFile(string filename);
	void CopyFromImageArray(const ImageArray& source);
	void AddFromImageArray(const ImageArray& source);
	
	void MultiplyBy(const ImageArray& source);
	void DivideBy(const ImageArray& source);
	void DivideBy(const ImageArray& source, const ImageArray& nonzero_mask);

	void ScaledBy(const T& factor);
	
	void SetValue(const T value);


	// data access
	// data access follow the order of x y z, x is the fastest changing dimension
	__inline T& operator[](int i) const { return _image[i]; }
	__inline T& operator()(int i) const { return _image[i]; }
	__inline T& operator()(int i, int j) const { return _image[j * _dim_x + i]; }
	__inline T& operator()(int i, int j, int k) const { return _image[(k * _dim_y + j) * _dim_x + i]; }


	T GetSum();
	size_t GetSize();
	T* _image;

protected:
	int _dim_x;
	int _dim_y;
	int _dim_z;

private:
	void _AllocateData();
	void _DeAllocateData();
	size_t _num_elem;
};

#endif
