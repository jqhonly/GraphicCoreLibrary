#pragma once
#include "stdafx.h"

//using namespace std;
using namespace GCL;
using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
//using namespace System::Drawing;
//using namespace System::Drawing::Imaging;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<std::string>(str)
#define MARSHAL_ARRAY(n_array, m_array) \
  auto m_array = gcnew array<float>(n_array.Size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array.Data, n_array.Size * sizeof(unsigned char));

namespace GCLSharp
{
	public ref class ColorTransSharp
	{
		ColorTransNative * nativemodel;
		int width = NULL;
		int height = NULL;
		int deviceid = 0;
	public:
		ColorTransSharp(int _width, int _height, int _deviceid)
		{
			width = _width;
			height = _height;
			deviceid = _deviceid;
			nativemodel = new ColorTransNative(width, height, deviceid);
		}
		~ColorTransSharp()
		{
			this->!ColorTransSharp();
		}

		!ColorTransSharp()
		{
			delete nativemodel;
			nativemodel = NULL;
		}

		int Managed_ColorTrans_YV12toARGB32(array<Byte>^ m_yv12, array<Byte>^ m_rgba32)
		{
			pin_ptr<unsigned char> pUnmanagedYv12 = &m_yv12[0];
			unsigned char* n_yv12 = pUnmanagedYv12;//unable to delete
			unsigned char* n_rgba32 = new unsigned char[width*height * 4];
			int result = nativemodel->Native_ColorTrans_YV12toARGB32(n_yv12, n_rgba32);
			//m_rgba32 = gcnew array<unsigned char>(width*height*4);
			/*pin_ptr<unsigned char> pUnmanagedRgba32 = &m_rgba32[0];
			memcpy(pUnmanagedRgba32, n_rgba32, width*height * 4 * sizeof(unsigned char));*/
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < width*height * 4; i++)
			{
				m_rgba32[i] = n_rgba32[i];
			}
			delete[] n_rgba32;
			return result;
		}

		int Managed_ColorTrans_YV12toARGB32_RetineX(array<Byte>^ m_yv12, array<Byte>^ m_rgba32)
		{
			pin_ptr<unsigned char> pUnmanagedYv12 = &m_yv12[0];
			unsigned char* n_yv12 = pUnmanagedYv12;//unable to delete
			unsigned char* n_rgba32 = new unsigned char[width*height * 4];
			int result = nativemodel->Native_ColorTrans_YV12toARGB32_RetineX(n_yv12, n_rgba32);
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < width*height * 4; i++)
			{
				m_rgba32[i] = n_rgba32[i];
			}
			delete[] n_rgba32;
			return result;
		}
	};
}