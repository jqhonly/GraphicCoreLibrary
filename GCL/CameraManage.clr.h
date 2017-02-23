#pragma once
#include "stdafx.h"

using namespace GCL;
using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<std::string>(str)

namespace GCLSharp
{
	public ref class CameraManageSharp
	{
		CameraManage *cameramanagenative;
	public:

		int width;
		int height;

		CameraManageSharp(System::String^ ip, short port, System::String^ user, System::String^ pwd)
		{
			cameramanagenative = new CameraManage(TO_NATIVE_STRING(ip), port, TO_NATIVE_STRING(user), TO_NATIVE_STRING(pwd));
		}

		~CameraManageSharp()
		{
			this->!CameraManageSharp();
		}

		!CameraManageSharp()
		{
			delete cameramanagenative;
			cameramanagenative = NULL;
		}

		bool login_managed()
		{
			return cameramanagenative->login_native();
		}
		bool play_managed(IntPtr display_hwnd)
		{
			return cameramanagenative->play_native((int)display_hwnd);
		}
		bool stop_managed()
		{
			return cameramanagenative->stop_native();
		}
		bool logout_managed()
		{
			return cameramanagenative->logout_native();
		}

		array<Byte>^ get_frame_managed()
		{
			unsigned char * temp = cameramanagenative->getFrame_native();
			array<Byte>^ rgba_data_output;// = gcnew array<Byte>(width * height * 4);
			
			if (temp!=nullptr)
			{
				height = cameramanagenative->Height;
				width = cameramanagenative->Width;
				rgba_data_output = gcnew array<Byte>(width * height * 4);
				int x = sizeof(temp);
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
				for (int i = 0; i < width*height * 4; i++)
				{
					rgba_data_output[i] = temp[i];
				}
				//delete[] temp;//unable to delete
				return rgba_data_output;
			}
			else
			{
				return nullptr;
			}
		}
	};
}